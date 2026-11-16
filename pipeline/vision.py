from pathlib import Path
import cv2
import pandas as pd


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


def analyze_visual_cues(video_path: str, start_sec: float, end_sec: float) -> float:
    """Checks 4 evenly spaced frames inside the candidate window for visible speaker faces."""
    if FACE_CASCADE is None or FACE_CASCADE.empty():
        return 1.0
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    s_frame = int(start_sec * video_fps)
    e_frame = int(end_sec * video_fps)
    if e_frame <= s_frame:
        cap.release()
        return 1.0

    step = max(1, (e_frame - s_frame) // 4)
    frames_checked, faces_detected = 0, 0

    for f_idx in range(s_frame, e_frame + 1, step):
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
    max_dur: float = 35.0,
) -> pd.DataFrame:
    if df_moments.empty:
        return df_moments

    starts, ends, durs, vis_scores, final_scores = [], [], [], [], []
    for _, row in df_moments.iterrows():
        ex_s = float(row.get("exchange_start", max(0.0, row["moment_x"] - 8.0)))
        ex_e = float(row.get("exchange_end", min(total_duration, row["moment_x"] + 8.0)))
        s = round(max(0.0, ex_s - buffer_y), 2)
        e = round(min(total_duration, ex_e + buffer_y), 2)
        if (e - s) > max_dur:
            e = round(s + max_dur, 2)

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
