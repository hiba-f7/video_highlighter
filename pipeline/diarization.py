import gc
import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from transformers import AutoFeatureExtractor, WavLMForXVector


def assign_speakers(df_sentences: pd.DataFrame, audio_path: str, n_speakers: int = 2) -> pd.DataFrame:
    """Extracts WavLM speaker embeddings using max 6s audio windows (prevents CUDA OOM), clusters via KMeans."""
    if df_sentences.empty:
        return df_sentences

    df_sentences = df_sentences.iloc[:80].copy().reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    model_name = "microsoft/wavlm-base-plus-sv"
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    wavlm = WavLMForXVector.from_pretrained(model_name).to(device)
    wavlm.eval()

    y, sr = librosa.load(audio_path, sr=16000)
    embeddings = []
    max_len = sr * 6  # 6 seconds max per embedding prevents O(N^2) attention VRAM spikes

    with torch.no_grad():
        for _, row in df_sentences.iterrows():
            s_idx = int(row["start"] * sr)
            e_idx = int(row["end"] * sr)
            clip = y[s_idx:e_idx][:max_len]
            if len(clip) < sr * 0.4:
                clip = np.pad(clip, (0, int(sr * 0.4) - len(clip)))
            inputs = extractor(clip, sampling_rate=16000, return_tensors="pt").to(device)
            emb = wavlm(**inputs).embeddings.squeeze().cpu().numpy()
            embeddings.append(emb)

    del wavlm, extractor
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    X = np.vstack(embeddings)
    k = min(n_speakers, len(X))
    if k > 1:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        df_sentences["speaker"] = [f"Speaker {lbl + 1}" for lbl in labels]
    else:
        df_sentences["speaker"] = "Speaker 1"

    return df_sentences
