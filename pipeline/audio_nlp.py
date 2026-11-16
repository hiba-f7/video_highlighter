import gc
import re
import librosa
import numpy as np
import pandas as pd
import torch
from faster_whisper import WhisperModel
from transformers import pipeline

def transcribe_and_merge(audio_path: str, device: str = "cpu") -> pd.DataFrame:
    """Loads Whisper-tiny, transcribes audio, and immediately frees RAM."""
    whisper_model = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
        num_workers=1,
    )
    segments, _ = whisper_model.transcribe(audio_path, vad_filter=True)
    raw_segs = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments if s.text.strip()
    ]
    del whisper_model
    gc.collect()

    # Merge fragments into complete sentences
    merged = []
    buf_text, buf_start, buf_end = "", None, None
    for seg in raw_segs:
        if buf_start is None:
            buf_start = seg["start"]
        buf_text = (buf_text + " " + seg["text"]).strip()
        buf_end = seg["end"]
        if re.search(r"[.?!]$", buf_text) or (buf_end - buf_start) >= 12.0:
            merged.append({"start": buf_start, "end": buf_end, "text": buf_text})
            buf_text, buf_start, buf_end = "", None, None
    if buf_text:
        merged.append({"start": buf_start, "end": buf_end, "text": buf_text})

    return pd.DataFrame(merged)

def detect_pitch_questions(df_sentences: pd.DataFrame, audio_path: str) -> pd.DataFrame:
    """Detects questions using punctuation and rising YIN F0 intonation."""
    if df_sentences.empty:
        return df_sentences
    y, sr = librosa.load(audio_path, sr=16000)
    is_q, pitch_scores = [], []
    for _, row in df_sentences.iterrows():
        q_punct = row["text"].strip().endswith("?")
        s_idx, e_idx = int(row["start"] * sr), int(row["end"] * sr)
        clip = y[s_idx:e_idx]
        p_rise = 0.0
        if len(clip) > sr * 0.5:
            f0 = librosa.yin(clip, fmin=75, fmax=400, sr=sr)
            f0 = f0[(f0 > 75) & (f0 < 400)]
            if len(f0) >= 6:
                mid = len(f0) // 2
                diff = np.mean(f0[mid:]) - np.mean(f0[:mid])
                p_rise = float(np.clip(diff / 30.0, 0.0, 1.0))
        is_q.append(q_punct or p_rise > 0.45)
        pitch_scores.append(round(1.0 if q_punct else p_rise, 2))
    df_sentences["is_question"] = is_q
    df_sentences["pitch_score"] = pitch_scores
    return df_sentences

def detect_highlight_moments(df_sentences: pd.DataFrame, selected_categories: list) -> pd.DataFrame:
    """Runs compact DistilBERT-MNLI on turn transitions and frees RAM immediately."""
    if df_sentences.empty or len(df_sentences) < 2:
        return pd.DataFrame()

    nli = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        device=-1,
    )
    labels = ["direct question and informative answer", "strong agreement and consensus", "disagreement and debate"]
    label_map = {
        "direct question and informative answer": "QA",
        "strong agreement and consensus": "Agreement",
        "disagreement and debate": "Disagreement",
    }

    moments = []
    # Cap candidate pairs to top 25 turn transitions so CPU inference finishes in ~10 seconds
    candidates = []
    for i in range(len(df_sentences) - 1):
        curr_row = df_sentences.iloc[i]
        next_row = df_sentences.iloc[i + 1]
        if curr_row["speaker"] != next_row["speaker"] or curr_row["is_question"]:
            candidates.append((curr_row, next_row))
    candidates = candidates[:25]

    for curr_row, next_row in candidates:
        pair_text = f"{curr_row['text']} {next_row['text']}"
        res = nli(pair_text, candidate_labels=labels)
        top_cat = label_map[res["labels"][0]]
        top_score = float(res["scores"][0])

        if top_cat == "QA" and curr_row["is_question"]:
            top_score = min(1.0, top_score + 0.15)

        if top_cat in selected_categories and top_score >= 0.45:
            moments.append({
                "moment_x": round((curr_row["end"] + next_row["start"]) / 2.0, 2),
                "category": top_cat,
                "nlp_score": round(top_score, 3),
                "trigger_text": curr_row["text"],
                "response_text": next_row["text"],
                "speakers": f"{curr_row['speaker']} → {next_row['speaker']}",
            })

    del nli
    gc.collect()
    return pd.DataFrame(moments)