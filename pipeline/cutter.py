import subprocess
from pathlib import Path
import pandas as pd


def filter_non_overlapping(df: pd.DataFrame, max_clips: int = 5, top_n: int = None) -> pd.DataFrame:
    if df.empty:
        return df

    limit = top_n if top_n is not None else max_clips
    score_col = "final_multimodal_score" if "final_multimodal_score" in df.columns else "nlp_score"
    df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)

    selected = []
    for _, row in df_sorted.iterrows():
        s, e = row["clip_start"], row["clip_end"]
        overlap = any(not (e <= sel["clip_start"] or s >= sel["clip_end"]) for sel in selected)
        if not overlap:
            selected.append(row)
        if len(selected) >= limit:
            break

    return pd.DataFrame(selected).reset_index(drop=True)


def extract_highlight_clips(video_path: str, df_highlights: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if df_highlights.empty:
        return df_highlights

    output_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = []

    for idx, row in df_highlights.iterrows():
        start_sec = float(row["clip_start"])
        duration_sec = float(row["clip_end"]) - start_sec
        cat = str(row.get("category", "clip")).lower()
        out_file = output_dir / f"highlight_{idx + 1}_{cat}.mp4"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", video_path,
            "-t", str(duration_sec),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            str(out_file),
            "-loglevel", "error",
        ]
        subprocess.run(cmd, check=True)
        clip_paths.append(str(out_file))

    df_out = df_highlights.copy()
    df_out["clip_path"] = clip_paths
    return df_out
