import subprocess
from pathlib import Path
import pandas as pd

def filter_non_overlapping(df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Selects top-ranked highlights while preventing overlapping time windows."""
    selected = []
    for _, row in df.iterrows():
        overlap = False
        for chosen in selected:
            if not (row["clip_end"] <= chosen["clip_start"] or row["clip_start"] >= chosen["clip_end"]):
                overlap = True
                break
        if not overlap:
            selected.append(row.to_dict())
        if len(selected) >= top_k:
            break
    return pd.DataFrame(selected)

def extract_highlight_clips(df_highlights: pd.DataFrame, video_path: str, output_dir: Path) -> pd.DataFrame:
    """Slices each highlight interval into browser-ready H.264/AAC MP4 files."""
    if df_highlights.empty:
        return df_highlights

    for cat in ["QA", "Agreement", "Disagreement"]:
        (output_dir / cat).mkdir(parents=True, exist_ok=True)

    clip_paths = []
    for idx, row in df_highlights.iterrows():
        filename = f"clip_{idx+1:02d}_{row['type']}_{int(row['clip_duration'])}s.mp4"
        out_path = output_dir / row["type"] / filename

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(row["clip_start"]),
            "-to", str(row["clip_end"]),
            "-i", video_path,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_path),
            "-loglevel", "error"
        ]
        subprocess.run(cmd, check=True)
        clip_paths.append(str(out_path))

    df_result = df_highlights.copy()
    df_result["clip_path"] = clip_paths
    return df_result