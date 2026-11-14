import warnings
import logging
import cv2
import pandas as pd
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_scenes(video_path: str, threshold: float = 27.0, min_scene_len: int = 15) -> list:
    """Detects visual scene/camera cuts using PySceneDetect ContentDetector."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.getLogger("pyscenedetect").setLevel(logging.ERROR)

        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        scenes = []
        for s_start, s_end in scene_list:
            start_sec = s_start.seconds if hasattr(s_start, "seconds") else s_start.get_seconds()
            end_sec = s_end.seconds if hasattr(s_end, "seconds") else s_end.get_seconds()
            scenes.append({"start": float(start_sec), "end": float(end_sec)})
    return scenes

def snap_to_scene_boundaries(moment_x: float, buffer_y: float, scenes: list, total_duration: float, min_dur: float = 30.0, max_dur: float = 60.0):
    """Snaps [x - y, x + y] to nearest camera cuts while enforcing 30s to 60s duration."""
    ideal_start = max(0.0, moment_x - buffer_y)
    ideal_end = min(total_duration, moment_x + buffer_y)

    snapped_start = ideal_start
    snapped_end = ideal_end

    if scenes:
        cut_starts = [s["start"] for s in scenes if s["start"] <= moment_x]
        if cut_starts:
            best_start = min(cut_starts, key=lambda t: abs(t - ideal_start))
            if abs(best_start - ideal_start) <= 5.0:
                snapped_start = best_start

        cut_ends = [s["end"] for s in scenes if s["end"] >= moment_x]
        if cut_ends:
            best_end = min(cut_ends, key=lambda t: abs(t - ideal_end))
            if abs(best_end - ideal_end) <= 5.0:
                snapped_end = best_end

    duration = snapped_end - snapped_start
    if duration < min_dur:
        needed = min_dur - duration
        snapped_start = max(0.0, snapped_start - needed / 2.0)
        snapped_end = min(total_duration, snapped_end + needed / 2.0)
    elif duration > max_dur:
        excess = duration - max_dur
        snapped_start += excess / 2.0
        snapped_end -= excess / 2.0

    return round(snapped_start, 2), round(snapped_end, 2)

def analyze_visual_cues(video_path: str, start_sec: float, end_sec: float, sample_fps: int = 1) -> float:
    """Calculates ratio of sampled frames containing clearly visible speaker faces."""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(video_fps / sample_fps))

    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_checked = 0
    faces_detected = 0
    current_frame = start_frame

    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (current_frame - start_frame) % frame_interval == 0:
            frames_checked += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40))
            if len(faces) > 0:
                faces_detected += 1
        current_frame += 1

    cap.release()
    score = (faces_detected / frames_checked) if frames_checked > 0 else 0.0
    return round(score, 2)

def enrich_with_vision(df_moments: pd.DataFrame, video_path: str, total_duration: float, buffer_y: float = 20.0) -> pd.DataFrame:
    """Applies scene snapping, visual face scoring, and multimodal ranking (70% NLP / 30% Vision)."""
    if df_moments.empty:
        return df_moments

    scenes = detect_scenes(video_path)
    results = []

    for _, row in df_moments.iterrows():
        c_start, c_end = snap_to_scene_boundaries(row["moment_x"], buffer_y, scenes, total_duration)
        vis_score = analyze_visual_cues(video_path, c_start, c_end)
        final_score = round(0.70 * row["nlp_score"] + 0.30 * vis_score, 3)

        item = row.to_dict()
        item["clip_start"] = c_start
        item["clip_end"] = c_end
        item["clip_duration"] = round(c_end - c_start, 2)
        item["visual_score"] = vis_score
        item["final_multimodal_score"] = final_score
        results.append(item)

    df_out = pd.DataFrame(results).sort_values(by="final_multimodal_score", ascending=False).reset_index(drop=True)
    return df_out