import subprocess
from pathlib import Path


def download_youtube_video(url: str, work_dir: Path) -> str:
    """Downloads a YouTube video as MP4 using cookie-free mobile/VR clients with yt-dlp fallback."""
    work_dir.mkdir(parents=True, exist_ok=True)
    video_path = work_dir / "input_video.mp4"

    if video_path.exists():
        video_path.unlink()

    clean_url = url.split("?si=")[0] if "?si=" in url else url

    for client_name in ["ANDROID_VR", "MWEB", "IOS", "ANDROID"]:
        try:
            from pytubefix import YouTube
            yt = YouTube(clean_url, client=client_name)
            stream = (
                yt.streams.filter(progressive=True, file_extension="mp4")
                .order_by("resolution")
                .desc()
                .first()
            )
            if stream:
                stream.download(output_path=str(work_dir), filename="input_video.mp4")
                if video_path.exists() and video_path.stat().st_size > 10000:
                    return str(video_path)
        except Exception:
            continue

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extractor-args", "youtube:player_client=tv_embedded,android_vr,mweb",
        "-f", "best[ext=mp4][height<=720]/best[ext=mp4]/best",
        "-o", str(video_path),
        clean_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and video_path.exists():
        return str(video_path)

    raise RuntimeError(
        "YouTube blocked this download request. Please upload an MP4 video directly in the 'Or Upload MP4 Clip' box."
    )


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
    """Returns exact video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())
