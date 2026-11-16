import gc
import re
import librosa
import numpy as np
import pandas as pd
import torch
from faster_whisper import WhisperModel
from transformers import pipeline

QUESTION_STARTERS = (
    "what", "why", "how", "when", "where", "who", "whom", "whose", "which",
    "do ", "does ", "did ", "is ", "are ", "was ", "were ", "can ", "could ",
    "would ", "will ", "should ", "have ", "has ", "had ", "isn't", "aren't",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "i wonder", "tell me",
)

AGREEMENT_CUES = (
    "i agree", "totally agree", "completely agree", "exactly", "precisely",
    "that's right", "that is right", "you're right", "you are right",
    "absolutely", "definitely", "for sure", "of course", "makes sense",
    "i think so too", "same here", "spot on", "true,", "yes,", "yeah,",
)

DISAGREEMENT_CUES = (
    "i disagree", "don't agree", "do not agree", "not true", "that's not",
    "that is not", "on the contrary", "no,", "but the thing is", "however,",
    "i don't think", "i do not think", "wrong", "mistaken", "push back",
)


def transcribe_and_merge(audio_path: str) -> pd.DataFrame:
    """Transcribes audio on GPU if CUDA 12 cuBLAS is available, with automatic int8 CPU fallback."""
    audio_array, _ = librosa.load(audio_path, sr=16000)
    max_samples = 16000 * 60 * 8
    if len(audio_array) > max_samples:
        audio_array = audio_array[:max_samples]

    use_gpu = torch.cuda.is_available()
    try:
        whisper_model = WhisperModel(
            "tiny",
            device="cuda" if use_gpu else "cpu",
            compute_type="float16" if use_gpu else "int8",
            cpu_threads=2,
        )
        segments, _ = whisper_model.transcribe(audio_array, vad_filter=True, beam_size=1)
        raw_segs = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments
            if s.text.strip()
        ]
    except RuntimeError:
        use_gpu = False
        whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=2)
        segments, _ = whisper_model.transcribe(audio_array, vad_filter=True, beam_size=1)
        raw_segs = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments
            if s.text.strip()
        ]

    del whisper_model
    if use_gpu:
        torch.cuda.empty_cache()
    gc.collect()

    merged = []
    buf_text, buf_start, buf_end = "", None, None
    for seg in raw_segs:
        if buf_start is None:
            buf_start = seg["start"]
        buf_text = (buf_text + " " + seg["text"]).strip()
        buf_end = seg["end"]
        if re.search(r"[.?!]$", buf_text) or (buf_end - buf_start) >= 14.0:
            merged.append({"start": buf_start, "end": buf_end, "text": buf_text})
            buf_text, buf_start, buf_end = "", None, None

    if buf_text:
        merged.append({"start": buf_start, "end": buf_end, "text": buf_text})

    return pd.DataFrame(merged)


def detect_pitch_questions(df_sentences: pd.DataFrame, audio_path: str) -> pd.DataFrame:
    """Fast lexical + rising pitch check (only runs YIN pitch on sentences without '?' or question starters)."""
    if df_sentences.empty:
        return df_sentences

    y, sr = librosa.load(audio_path, sr=16000)
    is_q, pitch_scores = [], []

    for _, row in df_sentences.iterrows():
        txt = row["text"].strip()
        low = txt.lower()
        q_punct = txt.endswith("?")
        q_lex = low.startswith(QUESTION_STARTERS) or " i wonder " in low

        if q_punct or q_lex:
            is_q.append(True)
            pitch_scores.append(1.0)
            continue

        s_idx, e_idx = int(row["start"] * sr), int(row["end"] * sr)
        clip = y[s_idx:e_idx]
        p_rise = 0.0
        if len(clip) > sr * 0.5:
            f0 = librosa.yin(clip, fmin=75, fmax=400, sr=sr, hop_length=512)
            f0 = f0[(f0 > 75) & (f0 < 400)]
            if len(f0) >= 6:
                mid = len(f0) // 2
                diff = np.mean(f0[mid:]) - np.mean(f0[:mid])
                p_rise = float(np.clip(diff / 30.0, 0.0, 1.0))

        is_q.append(bool(p_rise > 0.55))
        pitch_scores.append(round(p_rise, 2))

    df_sentences["is_question"] = is_q
    df_sentences["pitch_score"] = pitch_scores
    return df_sentences


def detect_highlight_moments(df_sentences: pd.DataFrame, selected_categories: list) -> pd.DataFrame:
    """Runs batched GPU DistilBERT-MNLI on cross-speaker exchanges in ~1–2 seconds."""
    if df_sentences.empty or len(df_sentences) < 2:
        return pd.DataFrame()

    turns = []
    for _, row in df_sentences.iterrows():
        if turns and turns[-1]["speaker"] == row["speaker"]:
            turns[-1]["end"] = row["end"]
            turns[-1]["text"] += " " + row["text"]
            turns[-1]["is_question"] = turns[-1]["is_question"] or row["is_question"]
        else:
            turns.append(dict(row))

    df_turns = pd.DataFrame(turns)
    if len(df_turns) < 2:
        return pd.DataFrame()

    pairs = []
    pair_texts = []
    for i in range(len(df_turns) - 1):
        t1 = df_turns.iloc[i]
        t2 = df_turns.iloc[i + 1]
        if t1["speaker"] == t2["speaker"] or (t2["end"] - t2["start"]) < 2.5:
            continue
        pairs.append((t1, t2))
        pair_texts.append(f"Speaker A: {t1['text']} | Speaker B: {t2['text']}")

    if not pairs:
        return pd.DataFrame()

    use_gpu = torch.cuda.is_available()
    nli = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        device=0 if use_gpu else -1,
    )
    labels = [
        "speaker agrees and confirms the other person's point",
        "speaker disagrees, challenges, or argues against the other person",
        "informative explanation answering a question",
    ]

    results = nli(pair_texts, candidate_labels=labels, batch_size=16)
    if isinstance(results, dict):
        results = [results]

    moments = []
    for (t1, t2), res in zip(pairs, results):
        scores_by_label = dict(zip(res["labels"], res["scores"]))
        agree_score = float(scores_by_label[labels[0]])
        disagree_score = float(scores_by_label[labels[1]])
        qa_score = float(scores_by_label[labels[2]])

        resp_low = t2["text"].lower()
        has_agree_cue = any(cue in resp_low for cue in AGREEMENT_CUES)
        has_disagree_cue = any(cue in resp_low for cue in DISAGREEMENT_CUES)

        category = None
        final_conf = 0.0

        if t1["is_question"] and not has_disagree_cue and qa_score >= 0.45:
            category = "QA"
            final_conf = round(min(0.99, qa_score + 0.20), 3)
        elif (has_agree_cue and agree_score >= 0.55) or agree_score >= 0.78:
            category = "Agreement"
            final_conf = round(min(0.99, agree_score + (0.15 if has_agree_cue else 0.0)), 3)
        elif (has_disagree_cue and disagree_score >= 0.60) or disagree_score >= 0.82:
            category = "Disagreement"
            final_conf = round(min(0.99, disagree_score + (0.10 if has_disagree_cue else 0.0)), 3)

        if category and category in selected_categories and final_conf >= 0.72:
            moments.append(
                {
                    "exchange_start": float(t1["start"]),
                    "exchange_end": float(t2["end"]),
                    "moment_x": round((t1["end"] + t2["start"]) / 2.0, 2),
                    "category": category,
                    "nlp_score": final_conf,
                    "trigger_text": t1["text"],
                    "response_text": t2["text"],
                    "speakers": f"{t1['speaker']} → {t2['speaker']}",
                }
            )

    del nli
    if use_gpu:
        torch.cuda.empty_cache()
    gc.collect()
    return pd.DataFrame(moments)
