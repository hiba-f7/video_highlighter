import gc
import re
import librosa
import numpy as np
import pandas as pd
import torch
from faster_whisper import WhisperModel
from transformers import pipeline

QUESTION_STARTERS = (
    "what ", "why ", "how ", "when ", "where ", "who ", "whom ", "whose ", "which ",
    "do you", "does ", "did you", "is it", "is there", "is that", "are you", "are we",
    "was ", "were ", "can you", "could you", "would you", "will you", "should ",
    "have you", "has ", "isn't ", "aren't ", "don't you", "doesn't ", "didn't ",
    "won't ", "wouldn't ", "tell me", "explain ", "i wonder",
)

QUESTION_SUBSTRINGS = (
    "do you think", "are you saying", "what do you", "why do you", "how do you",
    "can you explain", "what is your", "how can you", "don't you think", "isn't it true",
    "would you agree", "is that right", "what about", "why are you", "who is",
)

AGREEMENT_CUES = (
    "i agree", "totally agree", "completely agree", "exactly", "precisely",
    "that's right", "that is right", "you're right", "you are right",
    "absolutely", "definitely", "for sure", "of course", "makes sense",
    "i think so too", "same here", "spot on", "fair enough", "that's true",
    "that is true", "yes, i", "yes, that", "yeah, i", "certainly",
)

DIRECT_REBUTTAL_STARTERS = (
    "i disagree", "i don't agree", "i do not agree", "that's not true",
    "that is not true", "no, that's", "no, it's not", "no, i don't",
    "that's false", "that's wrong", "you're wrong", "not at all", "on the contrary",
)

DISAGREEMENT_CUES = (
    "i disagree", "don't agree", "do not agree", "not true", "that's not",
    "that is not", "on the contrary", "i don't think", "i do not think",
    "wrong", "mistaken", "push back", "not at all", "false", "ridiculous",
    "absurd", "i reject", "no, ", "but that's", "however,",
)

INTRO_NOISE = (
    "welcome to", "joining me", "good evening", "good morning",
    "thanks for watching", "subscribe", "see you next time",
)


def transcribe_and_merge(audio_path: str) -> pd.DataFrame:
    """Transcribes audio with Whisper-tiny (int8 CPU mode) and splits strictly at sentence punctuation."""
    audio_array, _ = librosa.load(audio_path, sr=16000)
    max_samples = 16000 * 60 * 12
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
        if re.search(r"[.?!]$", buf_text) or (buf_end - buf_start) >= 18.0:
            merged.append({"start": buf_start, "end": buf_end, "text": buf_text})
            buf_text, buf_start, buf_end = "", None, None

    if buf_text:
        merged.append({"start": buf_start, "end": buf_end, "text": buf_text})

    return pd.DataFrame(merged)


def detect_pitch_questions(df_sentences: pd.DataFrame, audio_path: str) -> pd.DataFrame:
    """Strictly identifies real questions via '?' punctuation or explicit interrogative grammar."""
    if df_sentences.empty:
        return df_sentences

    is_q, pitch_scores = [], []
    for _, row in df_sentences.iterrows():
        txt = row["text"].strip()
        low = txt.lower()
        q_punct = "?" in txt
        q_lex = low.startswith(QUESTION_STARTERS) or any(k in low for k in QUESTION_SUBSTRINGS)

        if q_punct or q_lex:
            is_q.append(True)
            pitch_scores.append(1.0)
        else:
            is_q.append(False)
            pitch_scores.append(0.0)

    df_sentences["is_question"] = is_q
    df_sentences["pitch_score"] = pitch_scores
    return df_sentences


def detect_highlight_moments(df_sentences: pd.DataFrame, selected_categories: list) -> pd.DataFrame:
    """Separates Q&A (question + explanation), Agreement (consensus), and Disagreement (statement challenge/rebuttal)."""
    if df_sentences.empty or len(df_sentences) < 2:
        return pd.DataFrame()

    turns = []
    for _, row in df_sentences.iterrows():
        if (
            turns
            and turns[-1]["speaker"] == row["speaker"]
            and (float(row["start"]) - float(turns[-1]["end"])) <= 3.5
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

        if (t2["end"] - t2["start"]) < 3.0:
            continue
        if any(phrase in t2["text"].lower() for phrase in INTRO_NOISE) or any(
            phrase in t1["text"].lower() for phrase in INTRO_NOISE
        ):
            continue

        # Take the last 1-2 sentences of Speaker 1 as the exact trigger
        t1_sents = t1["sentences"][-2:]
        ex_start = float(t1_sents[0]["start"])

        # Core response from Speaker 2 (used for clean NLI classification)
        core_t2_sents = []
        for s in t2["sentences"]:
            if (float(s["end"]) - ex_start) > 56.0 and core_t2_sents:
                break
            core_t2_sents.append(s)
            if (float(s["end"]) - ex_start) >= 32.0:
                break

        core_trig_text = " ".join(s["text"] for s in t1_sents)
        core_resp_text = " ".join(s["text"] for s in core_t2_sents)

        # Check if the final sentence(s) of Speaker 1 actually ask a question
        trig_low = core_trig_text.lower().strip()
        trig_is_q = (
            "?" in core_trig_text
            or t1["sentences"][-1]["is_question"]
            or trig_low.startswith(QUESTION_STARTERS)
            or any(k in trig_low for k in QUESTION_SUBSTRINGS)
        )

        # Extend clip window with complete sentences if needed to hit 30s-58s
        display_t2_sents = list(core_t2_sents)
        curr_turn_idx = i + 2
        while (float(display_t2_sents[-1]["end"]) - ex_start) < 28.0 and curr_turn_idx < len(turns):
            next_t = turns[curr_turn_idx]
            if (float(next_t["start"]) - float(display_t2_sents[-1]["end"])) > 4.5:
                break
            for s in next_t["sentences"]:
                if (float(s["end"]) - ex_start) > 58.0:
                    break
                display_t2_sents.append(s)
                if (float(s["end"]) - ex_start) >= 32.0:
                    break
            curr_turn_idx += 1

        ex_end = float(display_t2_sents[-1]["end"])
        if (ex_end - ex_start) < 14.0:
            continue

        display_resp_text = " ".join(s["text"] for s in display_t2_sents)
        pairs.append(
            (
                t1["speaker"],
                t2["speaker"],
                ex_start,
                ex_end,
                core_trig_text,
                core_resp_text,
                display_resp_text,
                trig_is_q,
            )
        )
        pair_texts.append(f"Speaker 1: {core_trig_text} | Speaker 2: {core_resp_text}")

    if not pairs:
        return pd.DataFrame()

    use_gpu = torch.cuda.is_available()
    nli = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        device=0 if use_gpu else -1,
    )
    labels = [
        "speaker 2 agrees with, confirms, or supports speaker 1",
        "speaker 2 disagrees with, contradicts, or argues against speaker 1",
        "speaker 1 asks an interview question and speaker 2 answers it",
    ]

    results = nli(pair_texts, candidate_labels=labels, batch_size=16)
    if isinstance(results, dict):
        results = [results]

    moments = []
    for (spk1, spk2, ex_start, ex_end, trig_text, core_resp, display_resp, trig_is_q), res in zip(pairs, results):
        scores = dict(zip(res["labels"], res["scores"]))
        agree_score = float(scores[labels[0]])
        disagree_score = float(scores[labels[1]])
        qa_score = float(scores[labels[2]])

        resp_low = core_resp.lower().strip()
        has_agree_cue = any(cue in resp_low for cue in AGREEMENT_CUES)
        has_direct_rebuttal = resp_low.startswith(DIRECT_REBUTTAL_STARTERS) or any(
            cue in resp_low[:80] for cue in DIRECT_REBUTTAL_STARTERS
        )
        has_disagree_cue = any(cue in resp_low for cue in DISAGREEMENT_CUES)

        category = None
        final_conf = 0.0

        # RULE 1: Explicit Agreement ("I agree", "That's right", "Exactly", "Fair enough")
        if has_agree_cue or (not trig_is_q and agree_score >= 0.52):
            category = "Agreement"
            final_conf = round(min(0.99, max(0.65, agree_score + (0.20 if has_agree_cue else 0.0))), 3)

        # RULE 2: Trigger is a REAL Question -> Always QA unless the guest opens with a direct rebuttal ("I disagree", "That's not true")
        elif trig_is_q and not has_direct_rebuttal:
            category = "QA"
            final_conf = round(min(0.99, max(0.68, qa_score + 0.22)), 3)

        # RULE 3: Disagreement -> Either a non-question challenge/argument OR a question met with an explicit direct rebuttal
        elif has_direct_rebuttal or (not trig_is_q and (has_disagree_cue or disagree_score >= 0.45)):
            category = "Disagreement"
            final_conf = round(min(0.99, max(0.62, disagree_score + (0.15 if has_disagree_cue else 0.0))), 3)

        if category and category in selected_categories and final_conf >= 0.50:
            moments.append(
                {
                    "exchange_start": ex_start,
                    "exchange_end": ex_end,
                    "moment_x": round((ex_start + ex_end) / 2.0, 2),
                    "category": category,
                    "nlp_score": final_conf,
                    "trigger_text": trig_text,
                    "response_text": display_resp,
                    "speakers": f"{spk1} → {spk2}",
                }
            )

    del nli
    if use_gpu:
        torch.cuda.empty_cache()
    gc.collect()
    return pd.DataFrame(moments)
