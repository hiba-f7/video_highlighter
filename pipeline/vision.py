from pathlib import Path
import cv2
import pandas as pd
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def _load_face_cascade():
    if not hasattr(cv2, "CascadeClassifier"):
        return None
    candidates = []
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    cv2_dir = Path(cv2.__file__).resolve().parent
    candidates.extend([
        cv2_dir / "data" / "haarcascade_frontalface_default.xml",
        Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
    ])
    for path in candidates:
        if path.exists():
            clf = cv2.CascadeClassifier(str(path))
            if not clf.empty():
                return clf
    return None


FACE_CASCADE = _load_face_cascade()


def detect_scenes(video_path: str, threshold: float = 27.0, min_scene_len: int = 15) -> list:
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    scene_manager.detect_scenes(video, frame_skip=2)
    return [(round(s.get_seconds(), 2), round(e.get_seconds(), 2)) for s, e in scene_manager.get_scene_list()]


def snap_exchange_boundaries(
    ex_start: float,
    ex_end: float,
    total_duration: float,
    scene_boundaries: list,
    pad_sec: float = 1.5,
    max_dur: float = 35.0,
) -> tuple:
    raw_start = max(0.0, ex_start - pad_sec)
    raw_end = min(total_duration, ex_end + pad_sec)

    cuts = sorted({c for pair in scene_boundaries for c in pair})
    if cuts:
        s_cands = [c for c in cuts if abs(c - raw_start) <= 2.0 and c <= ex_start]
        if s_cands:
            raw_start = min(s_cands, key=lambda c: abs(c - raw_start))

        e_cands = [c for c in cuts if abs(c - raw_end) <= 2.0 and c >= ex_end]
        if e_cands:
            raw_end = min(e_cands, key=lambda c: abs(c - raw_end))

    if (raw_end - raw_start) > max_dur:
        raw_end = round(raw_start + max_dur, 2)

    return round(raw_start, 2), round(raw_end, 2)


def analyze_visual_cues(video_path: str, start_sec: float, end_sec: float, sample_fps: float = 0.5) -> float:
    if FACE_CASCADE is None or FACE_CASCADE.empty():
        return 1.0
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(video_fps / sample_fps))
    frames_checked, faces_detected = 0, 0

    for f_idx in range(int(start_sec * video_fps), int(end_sec * video_fps) + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames_checked += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40))
        if len(faces) > 0:
            faces_detected += 1

    cap.release()
    return round((faces_detected / frames_checked) if frames_checked > 0 else 0.0, 2)


def enrich_with_vision(
    df_moments: pd.DataFrame,
    video_path: str,
    total_duration: float,
    buffer_y: float = 1.5,
) -> pd.DataFrame:
    if df_moments.empty:
        return df_moments

    scenes = detect_scenes(video_path)
    starts, ends, durs, vis_scores, final_scores = [], [], [], [], []

    for _, row in df_moments.iterrows():
        ex_s = row.get("exchange_start", max(0.0, row["moment_x"] - 8.0))
        ex_e = row.get("exchange_end", min(total_duration, row["moment_x"] + 8.0))
        s, e = snap_exchange_boundaries(ex_s, ex_e, total_duration, scenes, pad_sec=min(buffer_y, 3.0))
        v_score = analyze_visual_cues(video_path, s, e)
        fused = round(0.80 * float(row["nlp_score"]) + 0.20 * v_score, 3)

        starts.append(s)
        ends.append(e)
        durs.append(round(e - s, 2))
        vis_scores.append(v_score)
        final_scores.append(fused)

    df_out = df_moments.copy()
    df_out["clip_start"] = starts
    df_out["clip_end"] = ends
    df_out["clip_duration"] = durs
    df_out["visual_score"] = vis_scores
    df_out["final_multimodal_score"] = final_scores
    return df_out
