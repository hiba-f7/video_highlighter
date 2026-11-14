import subprocess
from pathlib import Path

def download_youtube_video(url: str, work_dir: Path) -> str:
    """Downloads a YouTube video as MP4 (up to 720p) with cloud-friendly client fallbacks."""
    work_dir.mkdir(parents=True, exist_ok=True)
    video_path = work_dir / "input_video.mp4"

    if video_path.exists():
        video_path.unlink()

    # Strip share tracking parameters (?si=...) if present
    clean_url = url.split("?si=")[0] if "?si=" in url else url

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extractor-args", "youtube:player_client=android,ios,web",
        "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", str(video_path),
        clean_url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not video_path.exists():
        err_msg = (result.stderr or result.stdout or "Unknown yt-dlp error").strip().splitlines()[-1]
        raise RuntimeError(
            f"YouTube download blocked or failed on cloud server ({err_msg}). "
            "Please use 'Upload MP4 File' in the sidebar or run locally."
        )

    return str(video_path)

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
