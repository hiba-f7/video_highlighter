from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def _load_face_cascade():
    """Safely locates haarcascade_frontalface_default.xml across Windows, Linux, and headless OpenCV."""
    if not hasattr(cv2, "CascadeClassifier"):
        return None

    candidates = []
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")

    cv2_dir = Path(cv2.__file__).resolve().parent
    candidates.extend([
        cv2_dir / "data" / "haarcascade_frontalface_default.xml",
        Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
        Path("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"),
    ])

    for path in candidates:
        if path.exists():
            clf = cv2.CascadeClassifier(str(path))
            if not clf.empty():
                return clf
    return None


FACE_CASCADE = _load_face_cascade()


def detect_scenes(video_path: str, threshold: float = 27.0, min_scene_len: int = 15) -> list:
    """Detects visual scene/camera cuts using PySceneDetect ContentDetector."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    scene_manager.detect_scenes(video, frame_skip=2)  # Skip every 2 frames for 3x faster scene detection
    scene_list = scene_manager.get_scene_list()

    boundaries = []
    for start_tc, end_tc in scene_list:
        boundaries.append((round(start_tc.get_seconds(), 2), round(end_tc.get_seconds(), 2)))
    return boundaries


def snap_to_scene_boundaries(
    moment_x: float,
    total_duration: float,
    scene_boundaries: list,
    buffer_y: float = 20.0,
    min_dur: float = 30.0,
    max_dur: float = 60.0,
) -> tuple:
    """Snaps [X - Y, X + Y] window to nearest camera transitions within 30–60s bounds."""
    raw_start = max(0.0, moment_x - buffer_y)
    raw_end = min(total_duration, moment_x + buffer_y)

    cuts = sorted({c for pair in scene_boundaries for c in pair})
    if cuts:
        start_candidates = [c for c in cuts if abs(c - raw_start) <= 4.0 and c < moment_x]
        if start_candidates:
            raw_start = min(start_candidates, key=lambda c: abs(c - raw_start))

        end_candidates = [c for c in cuts if abs(c - raw_end) <= 4.0 and c > moment_x]
        if end_candidates:
            raw_end = min(end_candidates, key=lambda c: abs(c - raw_end))

    dur = raw_end - raw_start
    if dur < min_dur:
        pad = (min_dur - dur) / 2.0
        raw_start = max(0.0, raw_start - pad)
        raw_end = min(total_duration, raw_end + pad)
        if (raw_end - raw_start) < min_dur:
            if raw_start == 0.0:
                raw_end = min(total_duration, min_dur)
            else:
                raw_start = max(0.0, total_duration - min_dur)
    elif dur > max_dur:
        excess = (dur - max_dur) / 2.0
        raw_start += excess
        raw_end -= excess

    return round(raw_start, 2), round(raw_end, 2)


def analyze_visual_cues(video_path: str, start_sec: float, end_sec: float, sample_fps: float = 0.5) -> float:
    """Fast-seeks 1 frame every 2 seconds to verify visible speaker faces."""
    if FACE_CASCADE is None or FACE_CASCADE.empty():
        return 1.0

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(video_fps / sample_fps))

    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)

    frames_checked = 0
    faces_detected = 0

    for f_idx in range(start_frame, end_frame + 1, step):
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
    score = (faces_detected / frames_checked) if frames_checked > 0 else 0.0
    return round(score, 2)


def enrich_with_vision(
    df_moments: pd.DataFrame,
    video_path: str,
    total_duration: float,
    buffer_y: float = 20.0,
) -> pd.DataFrame:
    """Snaps candidate moments to scene boundaries and computes fused multimodal scores."""
    if df_moments.empty:
        return df_moments

    scenes = detect_scenes(video_path)
    starts, ends, durs, vis_scores, final_scores = [], [], [], [], []

    for _, row in df_moments.iterrows():
        s, e = snap_to_scene_boundaries(row["moment_x"], total_duration, scenes, buffer_y=buffer_y)
        v_score = analyze_visual_cues(video_path, s, e)
        fused = round(0.75 * float(row["nlp_score"]) + 0.25 * v_score, 3)

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
