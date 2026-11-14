import torch
import numpy as np
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from transformers import AutoFeatureExtractor, WavLMForXVector

def load_wavlm_model(device: str = "cpu"):
    model_name = "microsoft/wavlm-base-plus-sv"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = WavLMForXVector.from_pretrained(model_name).to(device)
    model.eval()
    return feature_extractor, model

def assign_speakers(audio_path: str, segments: list, feature_extractor, model, device: str = "cpu") -> list:
    """Assigns SPEAKER_0 or SPEAKER_1 to each merged utterance using RMS peak windowing and KMeans."""
    if not segments:
        return segments

    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    embeddings = []
    valid_indices = []
    window_samples = int(3.0 * sr)  # 3-second peak speech window

    for idx, seg in enumerate(segments):
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = wav[start_sample:end_sample]

        if len(chunk) < int(0.4 * sr):
            continue

        # Find highest RMS energy 3s sub-window to avoid background music bleed
        if len(chunk) > window_samples:
            step = int(0.5 * sr)
            best_rms = -1.0
            best_sub = chunk[:window_samples]
            for pos in range(0, len(chunk) - window_samples + 1, step):
                sub = chunk[pos:pos + window_samples]
                rms = np.sqrt(np.mean(sub ** 2))
                if rms > best_rms:
                    best_rms = rms
                    best_sub = sub
            chunk = best_sub

        inputs = feature_extractor(chunk, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = model(**inputs).embeddings.cpu().numpy()[0]

        embeddings.append(emb)
        valid_indices.append(idx)

    if len(embeddings) < 2:
        for seg in segments:
            seg["speaker"] = "SPEAKER_0"
        return segments

    # L2 normalize embeddings before KMeans clustering into 2 speakers
    norm_embeddings = normalize(np.vstack(embeddings))
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(norm_embeddings)

    idx_to_label = {valid_indices[i]: f"SPEAKER_{labels[i]}" for i in range(len(valid_indices))}

    # Propagate labels (fill short skipped segments with nearest speaker)
    last_speaker = "SPEAKER_0"
    for i, seg in enumerate(segments):
        if i in idx_to_label:
            last_speaker = idx_to_label[i]
        seg["speaker"] = last_speaker

    return segments