import gc
import re
import librosa
import numpy as np
import pandas as pd
import torch
from faster_whisper import WhisperModel
from transformers import pipeline

QUESTION_PHRASES = (
    "what", "why", "how", "when", "where", "who", "whom", "whose", "which",
    "do ", "does ", "did ", "is ", "are ", "was ", "were ", "can ", "could ",
    "would ", "will ", "should ", "have ", "has ", "had ", "isn't", "aren't",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "i wonder", "tell me",
    "so you", "you're saying", "are you", "do you", "you say", "you argue",
)

AGREEMENT_CUES = (
    "i agree", "totally agree", "completely agree", "exactly", "precisely",
    "that's right", "that is right", "you're right", "you are right",
    "absolutely", "definitely", "for sure", "of course", "makes sense",
    "i think so too", "same here", "spot on", "fair enough",
)

DISAGREEMENT_CUES = (
    "i disagree", "don't agree", "do not agree", "not true", "that's not",
    "that is not", "on the contrary", "i don't think", "i do not think",
    "wrong", "mistaken", "push back", "not at all", "it's not political",
)

INTRO_NOISE = (
    "welcome to", "joining me", "good evening", "good morning",
    "thanks for watching", "subscribe", "see you next time",
)


def transcribe_and_merge(audio_path: str) -> pd.DataFrame:
    """Transcribes audio with Whisper-tiny (int8 CPU mode) and splits strictly at sentence punctuation."""
    audio_array, _ = librosa.load(audio_path, sr=16000)
    max_samples = 16000 * 60 * 10
    if len(audio_array) > max_samples:
        audio_array = audio_array[:max_samples]

    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=2)
    segments, _ = whisper_model.transcribe(audio_array, vad_filter=True, beam_size=1)
    raw_segs = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments
        if s.text.strip()
    ]
    del whisper_model
    gc.collect()

    merged = []
    buf_text, buf_start, buf_end = "", None, None
    for seg in raw_segs:
        if buf_start is None:
            buf_start = seg["start"]
        buf_text = (buf_text + " " + seg["text"]).strip()
        buf_end = seg["end"]
        if re.search(r"[.?!]$", buf_text):
            merged.append({"start": buf_start, "end": buf_end, "text": buf_text})
            buf_text, buf_start, buf_end = "", None, None

    if buf_text:
        merged.append({"start": buf_start, "end": buf_end, "text": buf_text})

    return pd.DataFrame(merged)


def detect_pitch_questions(df_sentences: pd.DataFrame, audio_path: str) -> pd.DataFrame:
    """Detects questions using punctuation, interrogative phrasing, and rising YIN F0 pitch."""
    if df_sentences.empty:
        return df_sentences

    y, sr = librosa.load(audio_path, sr=16000)
    is_q, pitch_scores = [], []

    for _, row in df_sentences.iterrows():
        txt = row["text"].strip()
        low = txt.lower()
        q_punct = "?" in txt
        q_lex = low.startswith(QUESTION_PHRASES) or any(
            k in low for k in ("i wonder", "why you", "why are", "how do", "what do", "do you", "are you", "isn't it")
        )

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
                p_rise = float(np.clip(diff / 25.0, 0.0, 1.0))

        is_q.append(bool(p_rise > 0.42))
        pitch_scores.append(round(p_rise, 2))

    df_sentences["is_question"] = is_q
    df_sentences["pitch_score"] = pitch_scores
    return df_sentences


def detect_highlight_moments(df_sentences: pd.DataFrame, selected_categories: list) -> pd.DataFrame:
    """Builds 30s–60s complete-sentence speaker exchanges and rejects music gaps/intros."""
    if df_sentences.empty or len(df_sentences) < 2:
        return pd.DataFrame()

    # Merge consecutive sentences by same speaker ONLY if pause between them <= 3.0s
    turns = []
    for _, row in df_sentences.iterrows():
        if (
            turns
            and turns[-1]["speaker"] == row["speaker"]
            and (float(row["start"]) - float(turns[-1]["end"])) <= 3.0
        ):
            turns[-1]["end"] = row["end"]
            turns[-1]["text"] += " " + row["text"]
            turns[-1]["is_question"] = turns[-1]["is_question"] or row["is_question"]
            turns[-1]["sentences"].append(dict(row))
        else:
            d = dict(row)
            d["sentences"] = [dict(row)]
            turns.append(d)

    if len(turns) < 2:
        return pd.DataFrame()

    pairs = []
    pair_texts = []
    for i in range(len(turns) - 1):
        t1 = turns[i]
        t2 = turns[i + 1]

        if t1["speaker"] == t2["speaker"]:
            continue

        silence_gap = float(t2["start"]) - float(t1["end"])
        if silence_gap > 4.5 or silence_gap < -0.5:
            continue

        if (t2["end"] - t2["start"]) < 4.0:
            continue
        if any(phrase in t2["text"].lower() for phrase in INTRO_NOISE) or any(
            phrase in t1["text"].lower() for phrase in INTRO_NOISE
        ):
            continue

        t1_sents = t1["sentences"][-2:]
        ex_start = float(t1_sents[0]["start"])

        t2_sents = []
        for s in t2["sentences"]:
            if (float(s["end"]) - ex_start) > 58.0 and t2_sents:
                break
            t2_sents.append(s)
            if (float(s["end"]) - ex_start) >= 32.0:
                break

        curr_turn_idx = i + 2
        while (float(t2_sents[-1]["end"]) - ex_start) < 30.0 and curr_turn_idx < len(turns):
            next_t = turns[curr_turn_idx]
            if (float(next_t["start"]) - float(t2_sents[-1]["end"])) > 4.5:
                break
            for s in next_t["sentences"]:
                if (float(s["end"]) - ex_start) > 58.0:
                    break
                t2_sents.append(s)
                if (float(s["end"]) - ex_start) >= 32.0:
                    break
            curr_turn_idx += 1

        ex_end = float(t2_sents[-1]["end"])
        dur = ex_end - ex_start
        if dur < 24.0 or dur > 60.0:
            continue

        trig_text = " ".join(s["text"] for s in t1_sents)
        resp_text = " ".join(s["text"] for s in t2_sents)
        trig_is_q = any(s["is_question"] for s in t1_sents) or ("?" in trig_text)

        pairs.append((t1["speaker"], t2["speaker"], ex_start, ex_end, trig_text, resp_text, trig_is_q))
        pair_texts.append(f"Speaker 1: {trig_text} | Speaker 2: {resp_text}")

    if not pairs:
        return pd.DataFrame()

    use_gpu = torch.cuda.is_available()
    nli = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        device=0 if use_gpu else -1,
    )
    labels = [
        "speaker 2 agrees, confirms, or supports speaker 1",
        "speaker 2 disagrees, rejects, or argues against speaker 1",
        "speaker 1 asks a question and speaker 2 explains or answers",
    ]

    results = nli(pair_texts, candidate_labels=labels, batch_size=16)
    if isinstance(results, dict):
        results = [results]

    moments = []
    for (spk1, spk2, ex_start, ex_end, trig_text, resp_text, trig_is_q), res in zip(pairs, results):
        scores = dict(zip(res["labels"], res["scores"]))
        agree_score = float(scores[labels[0]])
        disagree_score = float(scores[labels[1]])
        qa_score = float(scores[labels[2]])

        resp_opening = resp_text[:160].lower()
        has_agree_cue = any(cue in resp_opening for cue in AGREEMENT_CUES)
        has_disagree_cue = any(cue in resp_opening for cue in DISAGREEMENT_CUES)

        category = None
        final_conf = 0.0

        if has_agree_cue or (not trig_is_q and agree_score >= 0.58):
            category = "Agreement"
            final_conf = round(min(0.99, agree_score + (0.18 if has_agree_cue else 0.0)), 3)
        elif trig_is_q and not has_disagree_cue and disagree_score < 0.85:
            category = "QA"
            final_conf = round(min(0.99, max(0.62, qa_score + 0.22)), 3)
        elif has_disagree_cue or disagree_score >= 0.52:
            category = "Disagreement"
            final_conf = round(min(0.99, disagree_score + (0.12 if has_disagree_cue else 0.0)), 3)

        if category and category in selected_categories and final_conf >= 0.50:
            moments.append(
                {
                    "exchange_start": ex_start,
                    "exchange_end": ex_end,
                    "moment_x": round((ex_start + ex_end) / 2.0, 2),
                    "category": category,
                    "nlp_score": final_conf,
                    "trigger_text": trig_text,
                    "response_text": resp_text,
                    "speakers": f"{spk1} → {spk2}",
                }
            )

    del nli
    if use_gpu:
        torch.cuda.empty_cache()
    gc.collect()
    return pd.DataFrame(moments)
