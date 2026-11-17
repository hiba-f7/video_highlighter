import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

CACHE_DIR = Path(tempfile.gettempdir()) / "echoframe_yt_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _extract_video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else "default_video"


def download_youtube_video(url: str, work_dir: Path) -> str:
    """Downloads YouTube video with automatic local cache lookup, multi-client fallback, and public API mirrors."""
    work_dir.mkdir(parents=True, exist_ok=True)
    video_path = work_dir / "input_video.mp4"

    vid_id = _extract_video_id(url)
    cached_mp4 = CACHE_DIR / f"{vid_id}.mp4"
    colab_cache = Path("/content/yt_cache") / f"{vid_id}.mp4"

    # 1. Instant load if already cached in Colab or system temp
    for candidate in (colab_cache, cached_mp4):
        if candidate.exists() and candidate.stat().st_size > 50000:
            shutil.copy(candidate, video_path)
            return str(video_path)

    clean_url = f"https://www.youtube.com/watch?v={vid_id}" if vid_id != "default_video" else url.split("?si=")[0]

    # 2. Try pytubefix across multiple mobile/creator/embed clients
    for client_name in ["WEB_CREATOR", "IOS", "ANDROID_VR", "MWEB", "ANDROID", "TV_EMBED", "WEB_EMBED", "WEB"]:
        try:
            from pytubefix import YouTube
            yt = YouTube(clean_url, client=client_name, use_po_token=False)
            stream = (
                yt.streams.filter(progressive=True, file_extension="mp4")
                .order_by("resolution")
                .desc()
                .first()
            )
            if stream:
                stream.download(output_path=str(work_dir), filename="input_video.mp4")
                if video_path.exists() and video_path.stat().st_size > 50000:
                    shutil.copy(video_path, cached_mp4)
                    return str(video_path)
        except Exception:
            continue

    # 3. Try yt-dlp with multi-client and IPv4 forcing
    for player_arg in [
        "youtube:player_client=ios,web_creator",
        "youtube:player_client=android_vr,tv_embedded",
        "youtube:player_client=mweb,web_embedded",
    ]:
        cmd = [
            "yt-dlp",
            "--force-ipv4",
            "--no-check-certificates",
            "--no-playlist",
            "--extractor-args", player_arg,
            "-f", "best[ext=mp4][height<=720]/18/best[ext=mp4]/best",
            "-o", str(video_path),
            clean_url,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and video_path.exists() and video_path.stat().st_size > 50000:
            shutil.copy(video_path, cached_mp4)
            return str(video_path)

    # 4. Public Piped / Invidious API Mirror Fallback
    mirrors = [
        f"https://pipedapi.kavin.rocks/streams/{vid_id}",
        f"https://pipedapi.tokhmi.xyz/streams/{vid_id}",
        f"https://inv.tux.pizza/api/v1/videos/{vid_id}",
    ]
    for api_url in mirrors:
        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode())
            streams = data.get("videoStreams") or data.get("formatStreams") or []
            for st_item in streams:
                dl_url = st_item.get("url")
                if dl_url and (
                    "mp4" in str(st_item.get("mimeType", "")).lower()
                    or "mp4" in str(st_item.get("container", "")).lower()
                ):
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", dl_url, "-c", "copy", str(video_path), "-loglevel", "error"],
                        timeout=120,
                    )
                    if video_path.exists() and video_path.stat().st_size > 50000:
                        shutil.copy(video_path, cached_mp4)
                        return str(video_path)
        except Exception:
            continue

    import streamlit as st
    st.warning(
        "**YouTube blocked this cloud server IP** for direct URL downloading. "
        "Switch to **Upload MP4 Clip** in the top control bar, or reconnect your Colab runtime to get a fresh IP."
    )
    st.stop()


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
