import shutil
import tempfile
from pathlib import Path
import torch
import pandas as pd
import streamlit as st

from pipeline.downloader import download_youtube_video, extract_audio, get_video_duration
from pipeline.audio_nlp import load_asr_and_nli, transcribe_and_merge, detect_moments
from pipeline.diarization import load_wavlm_model, assign_speakers
from pipeline.vision import enrich_with_vision
from pipeline.cutter import filter_non_overlapping, extract_highlight_clips

st.set_page_config(
    page_title="EchoFrame | Multimodal Content & Reaction Analyzer",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS for gradient hero, colored category badges, and styled cards
st.markdown("""
<style>
    .hero-box {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #0F172A 100%);
        border: 1px solid #4338CA;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-sub {
        color: #CBD5E1;
        font-size: 1.02rem;
        margin-bottom: 14px;
    }
    .tech-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid #6366F1;
        color: #C7D2FE;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-right: 8px;
    }
    .step-card {
        background: #1E293B;
        border-top: 3px solid #06B6D4;
        border-radius: 10px;
        padding: 16px;
        height: 100%;
    }
    .badge-QA {
        background: rgba(139, 92, 246, 0.2);
        color: #C4B5FD;
        border: 1px solid #8B5CF6;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-Agreement {
        background: rgba(16, 185, 129, 0.2);
        color: #6EE7B7;
        border: 1px solid #10B981;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-Disagreement {
        background: rgba(244, 63, 94, 0.2);
        color: #FDA4AF;
        border: 1px solid #F43F5E;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .speaker-0 { color: #38BDF8; font-weight: 700; }
    .speaker-1 { color: #F472B6; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model, nli_classifier = load_asr_and_nli(device=device)
    wavlm_extractor, wavlm_model = load_wavlm_model(device=device)
    return device, whisper_model, nli_classifier, wavlm_extractor, wavlm_model

# Hero Banner
st.markdown("""
<div class="hero-box">
    <div class="hero-title">AI Interview & Podcast Highlight Studio</div>
    <div class="hero-sub">
        Multimodal video intelligence that automatically extracts 30–60s shareable clips of key 
        <b>Q&A</b>, <b>Agreements</b>, and <b>Disagreements</b> snapped cleanly to camera transitions.
    </div>
    <span class="tech-pill">Faster-Whisper ASR</span>
    <span class="tech-pill">WavLM + KMeans Diarization</span>
    <span class="tech-pill">BART-MNLI + YIN Pitch</span>
    <span class="tech-pill">PySceneDetect + OpenCV</span>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.header("1. Video Source")
    input_mode = st.radio("Select Input Method:", ["YouTube URL", "Upload MP4 File"])

    youtube_url = ""
    uploaded_file = None

    if input_mode == "YouTube URL":
        youtube_url = st.text_input("Paste Interview URL:", placeholder="https://www.youtube.com/watch?v=...")
    else:
        uploaded_file = st.file_uploader("Upload Interview Video", type=["mp4", "mov", "mkv"])

    st.divider()
    st.header("2. Extraction Parameters")
    buffer_y = st.slider("Context Buffer Window (s)", min_value=15, max_value=30, value=20)
    top_k = st.slider("Max Clips to Generate", min_value=1, max_value=10, value=5)
    selected_types = st.multiselect(
        "Highlight Categories",
        options=["QA", "Agreement", "Disagreement"],
        default=["QA", "Agreement", "Disagreement"]
    )

    st.divider()
    run_btn = st.button("Generate Highlights", type="primary", use_container_width=True)

# Pipeline Execution
if run_btn:
    if input_mode == "YouTube URL" and not youtube_url.strip():
        st.error("Please enter a valid YouTube URL in the sidebar.")
        st.stop()
    if input_mode == "Upload MP4 File" and uploaded_file is None:
        st.error("Please upload an MP4 video file in the sidebar.")
        st.stop()

    if "work_dir" in st.session_state and Path(st.session_state["work_dir"]).exists():
        shutil.rmtree(st.session_state["work_dir"], ignore_errors=True)

    work_dir = Path(tempfile.mkdtemp(prefix="highlighter_"))
    st.session_state["work_dir"] = str(work_dir)

    with st.status("Running Multimodal Highlight Pipeline...", expanded=True) as status:
        st.write("Loading neural models (Faster-Whisper, WavLM, BART-large-MNLI)...")
        device, whisper_model, nli_classifier, wavlm_extractor, wavlm_model = get_models()

        if input_mode == "YouTube URL":
            st.write("Downloading interview video via yt-dlp...")
            video_path = download_youtube_video(youtube_url.strip(), work_dir)
        else:
            st.write("Preparing uploaded video...")
            video_path = str(work_dir / "uploaded_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

        total_duration = get_video_duration(video_path)

        st.write("Extracting 16kHz mono audio & transcribing dialogue...")
        audio_path = extract_audio(video_path, work_dir)
        segments = transcribe_and_merge(audio_path, whisper_model)

        st.write("Separating speakers with WavLM embeddings & KMeans...")
        segments = assign_speakers(audio_path, segments, wavlm_extractor, wavlm_model, device=device)
        st.session_state["segments"] = segments

        st.write("Scoring Q&A pitch contours & NLI agreement/disagreement turns...")
        df_moments = detect_moments(segments, audio_path, nli_classifier)

        if not df_moments.empty and selected_types:
            df_moments = df_moments[df_moments["type"].isin(selected_types)].reset_index(drop=True)

        if df_moments.empty:
            status.update(label="No matching highlights found.", state="error")
            st.warning("No conversation turns matched your selected filters.")
            st.stop()

        st.write("Snapping boundaries to PySceneDetect cuts & verifying speaker faces...")
        df_enriched = enrich_with_vision(df_moments, video_path, total_duration, buffer_y=float(buffer_y))
        df_top = filter_non_overlapping(df_enriched, top_k=top_k)

        st.write("Rendering 30–60s H.264 highlight clips with FFmpeg...")
        df_final = extract_highlight_clips(df_top, video_path, work_dir / "outputs")
        st.session_state["df_final"] = df_final

        status.update(label="Highlight Generation Complete!", state="complete", expanded=False)

# Display Results OR Rich Pre-Run Overview
if "df_final" in st.session_state and not st.session_state["df_final"].empty:
    df_final = st.session_state["df_final"]
    segments = st.session_state.get("segments", [])

    # KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clips Extracted", len(df_final))
    m2.metric("Avg Multimodal Score", f"{df_final['final_multimodal_score'].mean():.2f}")
    m3.metric("Total Highlight Time", f"{df_final['clip_duration'].sum():.0f}s")
    m4.metric("Top Category", df_final["type"].mode()[0])

    st.write("")
    tab_clips, tab_transcript, tab_analytics = st.tabs([
        "🎬 Highlight Clips",
        "🎙️ Full Diarized Transcript",
        "📊 Scores & Export"
    ])

    with tab_clips:
        for idx, row in df_final.iterrows():
            badge_cls = f"badge-{row['type']}"
            with st.container(border=True):
                c_vid, c_info = st.columns([1.25, 1])
                with c_vid:
                    st.video(row["clip_path"])
                with c_info:
                    st.markdown(
                        f"### Clip #{idx+1} &nbsp; <span class='{badge_cls}'>{row['type']}</span>",
                        unsafe_allow_html=True
                    )
                    st.caption(
                        f"⏱️ {row['clip_start']}s – {row['clip_end']}s ({row['clip_duration']}s)  |  "
                        f"🎯 Multimodal: **{row['final_multimodal_score']}** "
                        f"(NLP: {row['nlp_score']} · Vision: {row['visual_score']})"
                    )
                    st.info(row["dialogue"])

                    with open(row["clip_path"], "rb") as f:
                        st.download_button(
                            label=f"⬇️ Download Clip #{idx+1} (.mp4)",
                            data=f,
                            file_name=Path(row["clip_path"]).name,
                            mime="video/mp4",
                            key=f"dl_clip_{idx}",
                            use_container_width=True
                        )

    with tab_transcript:
        st.subheader("Speaker-Separated Dialogue")
        if segments:
            for seg in segments:
                spk_cls = "speaker-0" if seg["speaker"] == "SPEAKER_0" else "speaker-1"
                st.markdown(
                    f"`[{seg['start']:06.2f}s]` <span class='{spk_cls}'>{seg['speaker']}</span>: {seg['text']}",
                    unsafe_allow_html=True
                )
        else:
            st.info("Transcript will appear here after running the pipeline.")

    with tab_analytics:
        st.subheader("Category Distribution & Moment Table")
        st.bar_chart(df_final["type"].value_counts(), color="#6366F1")

        display_cols = ["type", "clip_start", "clip_end", "clip_duration", "nlp_score", "visual_score", "final_multimodal_score", "dialogue"]
        st.dataframe(df_final[display_cols], use_container_width=True)

        st.download_button(
            label="📥 Download Full Summary Report (CSV)",
            data=df_final[display_cols].to_csv(index=False).encode("utf-8"),
            file_name="interview_highlights_report.csv",
            mime="text/csv"
        )

else:
    # Rich Empty-State Guide before processing a video
    st.subheader("How the Multimodal Pipeline Works")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="step-card">
            <h4>1. Speech & Pitch</h4>
            <p style="color:#CBD5E1;font-size:0.9rem;">
                <b>Faster-Whisper</b> transcribes dialogue with VAD filtering, while <b>YIN F0</b> tracks rising intonation at question boundaries.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="step-card" style="border-top-color:#8B5CF6;">
            <h4>2. Speaker Clustering</h4>
            <p style="color:#CBD5E1;font-size:0.9rem;">
                <b>WavLM</b> extracts peak-energy voice embeddings and <b>KMeans</b> separates Host vs. Guest turns automatically.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="step-card" style="border-top-color:#10B981;">
            <h4>3. Semantic NLI</h4>
            <p style="color:#CBD5E1;font-size:0.9rem;">
                <b>BART-large-MNLI</b> evaluates cross-speaker responses to detect high-engagement <b>Q&A</b>, <b>Agreements</b>, and <b>Debates</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="step-card" style="border-top-color:#F43F5E;">
            <h4>4. Vision Snapping</h4>
            <p style="color:#CBD5E1;font-size:0.9rem;">
                <b>PySceneDetect</b> aligns cuts to camera switches and <b>OpenCV</b> verifies active speaker faces within 30–60s windows.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.info("👈 Paste a YouTube interview URL or upload an MP4 file in the sidebar and click **Generate Highlights** to begin.")