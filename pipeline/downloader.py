import subprocess
from pathlib import Path
import streamlit as st

def _get_cookie_file(work_dir: Path) -> Path | None:
    """Writes YOUTUBE_COOKIES from Streamlit Secrets to a temporary file if configured."""
    try:
        if "YOUTUBE_COOKIES" in st.secrets and st.secrets["YOUTUBE_COOKIES"].strip():
            cookie_path = work_dir / "cookies.txt"
            cookie_path.write_text(st.secrets["YOUTUBE_COOKIES"], encoding="utf-8")
            return cookie_path
    except Exception:
        pass
    return None

def download_youtube_video(url: str, work_dir: Path) -> str:
    """Downloads a YouTube video as MP4 using yt-dlp (with optional cookies) and pytubefix fallback."""
    work_dir.mkdir(parents=True, exist_ok=True)
    video_path = work_dir / "input_video.mp4"

    if video_path.exists():
        video_path.unlink()

    clean_url = url.split("?si=")[0] if "?si=" in url else url
    cookie_path = _get_cookie_file(work_dir)

    # 1. Primary attempt: yt-dlp with cloud-friendly clients & optional cookies
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--geo-bypass",
        "--extractor-args", "youtube:player_client=android,web",
        "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", str(video_path),
    ]
    if cookie_path:
        cmd.extend(["--cookies", str(cookie_path)])
    cmd.append(clean_url)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and video_path.exists():
        return str(video_path)

    yt_dlp_err = (result.stderr or result.stdout or "").strip()

    # 2. Secondary fallback: pytubefix (works on many cloud IPs without cookies)
    try:
        from pytubefix import YouTube
        yt = YouTube(clean_url, client="WEB")
        stream = (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
        )
        if stream:
            stream.download(output_path=str(work_dir), filename="input_video.mp4")
            if video_path.exists():
                return str(video_path)
    except Exception as pt_err:
        yt_dlp_err += f" | pytubefix fallback: {pt_err}"

    last_line = yt_dlp_err.splitlines()[-1] if yt_dlp_err else "Unknown download error"
    raise RuntimeError(f"YouTube download failed: {last_line}")

def extract_audio(video_path: str, work_dir: Path) -> str:
    """Extracts a 16kHz mono WAV audio track from the input video using FFmpeg."""
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / "audio.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path),
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)
    return str(audio_path)

def get_video_duration(video_path: str) -> float:
    """Returns the exact duration of the video in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())
