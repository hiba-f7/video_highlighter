import re
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
from faster_whisper import WhisperModel
from transformers import pipeline

QUESTION_STARTERS = (
    "who", "what", "when", "where", "why", "how", "is", "are", "was", "were",
    "do", "does", "did", "can", "could", "would", "should", "will", "have", "has"
)

AGREE_CUES = ["yes", "yeah", "exactly", "definitely", "absolutely", "i agree", "true", "right", "spot on", "for sure"]
DISAGREE_CUES = ["no", "actually", "i disagree", "not really", "however", "on the other hand", "i don't think", "but"]

def load_asr_and_nli(device: str = "cpu"):
    compute_type = "float16" if device == "cuda" else "int8"
    whisper_model = WhisperModel("small", device=device, compute_type=compute_type)
    nli_device = 0 if device == "cuda" else -1
    nli_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=nli_device)
    return whisper_model, nli_classifier

def transcribe_and_merge(audio_path: str, whisper_model) -> list:
    """Transcribes speech with VAD and merges broken sentence fragments."""
    raw_segments, _ = whisper_model.transcribe(audio_path, vad_filter=True)
    merged = []

    for seg in raw_segments:
        text = seg.text.strip()
        if not text:
            continue

        if not merged:
            merged.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
            continue

        prev = merged[-1]
        gap = seg.start - prev["end"]
        ends_with_punct = prev["text"].endswith((".", "?", "!"))

        if not ends_with_punct and gap <= 1.2:
            prev["text"] += " " + text
            prev["end"] = round(seg.end, 2)
        else:
            merged.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})

    return merged

def check_rising_pitch(wav: np.ndarray, sr: int, start_sec: float, end_sec: float) -> float:
    """Measures pitch rise in the final 0.8s of an utterance using YIN F0 tracking."""
    tail_start = max(start_sec, end_sec - 0.8)
    s_idx, e_idx = int(tail_start * sr), int(end_sec * sr)
    chunk = wav[s_idx:e_idx]
    if len(chunk) < int(0.2 * sr):
        return 0.0

    f0 = librosa.yin(chunk, fmin=75, fmax=400, sr=sr)
    f0 = f0[~np.isnan(f0)]
    if len(f0) < 4:
        return 0.0

    mid = len(f0) // 2
    first_half = np.median(f0[:mid])
    second_half = np.median(f0[mid:])
    return 1.0 if second_half > first_half * 1.05 else 0.0

def detect_moments(segments: list, audio_path: str, nli_classifier) -> pd.DataFrame:
    """Detects QA, Agreement, and Disagreement moments across cross-speaker turns."""
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    moments = []
    for i in range(len(segments) - 1):
        curr_seg = segments[i]
        next_seg = segments[i + 1]

        # Enforce cross-speaker turn transition
        if curr_seg["speaker"] == next_seg["speaker"]:
            continue

        turn_gap = next_seg["start"] - curr_seg["end"]
        if turn_gap > 4.0:
            continue

        text_a = curr_seg["text"]
        text_b = next_seg["text"]
        lower_a = text_a.lower().strip()
        lower_b = text_b.lower().strip()
        moment_x = round((curr_seg["end"] + next_seg["start"]) / 2.0, 2)

        # 1. Question-Answer Detection
        is_question_punct = text_a.endswith("?")
        first_word = re.sub(r"[^\w]", "", lower_a.split()[0]) if lower_a.split() else ""
        is_question_word = first_word in QUESTION_STARTERS
        pitch_rise = check_rising_pitch(wav, sr, curr_seg["start"], curr_seg["end"])

        if is_question_punct or (is_question_word and pitch_rise > 0):
            qa_conf = round(0.75 + (0.15 if is_question_punct else 0.0) + (0.10 * pitch_rise), 3)
            moments.append({
                "type": "QA",
                "moment_x": moment_x,
                "nlp_score": min(1.0, qa_conf),
                "speaker_a": curr_seg["speaker"],
                "speaker_b": next_seg["speaker"],
                "dialogue": f"[{curr_seg['speaker']}]: {text_a}  -->  [{next_seg['speaker']}]: {text_b}"
            })
            continue

        # 2. Agreement vs. Disagreement Detection via BART NLI + Lexical Cues
        pair_text = f"Speaker 1 said: '{text_a}'. Speaker 2 replied: '{text_b}'."
        candidate_labels = ["agreement and support", "disagreement and debate", "neutral continuation"]
        res = nli_classifier(pair_text, candidate_labels)
        top_label = res["labels"][0]
        top_score = res["scores"][0]

        has_agree_cue = any(cue in lower_b[:35] for cue in AGREE_CUES)
        has_disagree_cue = any(cue in lower_b[:35] for cue in DISAGREE_CUES)

        if top_label == "agreement and support" and (top_score >= 0.55 or has_agree_cue):
            final_nlp = min(1.0, round(top_score + (0.1 if has_agree_cue else 0.0), 3))
            moments.append({
                "type": "Agreement",
                "moment_x": moment_x,
                "nlp_score": final_nlp,
                "speaker_a": curr_seg["speaker"],
                "speaker_b": next_seg["speaker"],
                "dialogue": f"[{curr_seg['speaker']}]: {text_a}  -->  [{next_seg['speaker']}]: {text_b}"
            })
        elif top_label == "disagreement and debate" and (top_score >= 0.55 or has_disagree_cue):
            final_nlp = min(1.0, round(top_score + (0.1 if has_disagree_cue else 0.0), 3))
            moments.append({
                "type": "Disagreement",
                "moment_x": moment_x,
                "nlp_score": final_nlp,
                "speaker_a": curr_seg["speaker"],
                "speaker_b": next_seg["speaker"],
                "dialogue": f"[{curr_seg['speaker']}]: {text_a}  -->  [{next_seg['speaker']}]: {text_b}"
            })

    return pd.DataFrame(moments)