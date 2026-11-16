import tempfile
from pathlib import Path
import streamlit as st

from pipeline.downloader import download_youtube_video, extract_audio, get_video_duration
from pipeline.audio_nlp import transcribe_and_merge, detect_pitch_questions, detect_highlight_moments
from pipeline.diarization import assign_speakers
from pipeline.vision import enrich_with_vision
from pipeline.cutter import filter_non_overlapping, extract_highlight_clips

st.set_page_config(
    page_title="EchoFrame | Multimodal Content & Reaction Analyzer",
    page_icon="🔍",
    layout="wide",
)

# Custom Dark Creator Studio Styling
st.markdown(
    """
    <style>
    .hero-card {
        background: linear-gradient(135deg, #131B2E 0%, #1E293B 100%);
        border: 1px solid #2E3A59;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 6px;
    }
    .hero-sub {
        font-size: 0.98rem;
        color: #94A3B8;
        margin-bottom: 14px;
    }
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .tech-pill {
        background-color: rgba(99, 102, 241, 0.15);
        border: 1px solid #6366F1;
        color: #A5B4FC;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-qa {
        background-color: rgba(56, 189, 248, 0.18);
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-agree {
        background-color: rgba(52, 211, 153, 0.18);
        color: #34D399;
        border: 1px solid #34D399;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-disagree {
        background-color: rgba(251, 113, 133, 0.18);
        color: #FB7185;
        border: 1px solid #FB7185;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero Banner
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">EchoFrame: Multimodal Content & Reaction Analyzer</div>
        <div class="hero-sub">
            AI-driven media intelligence for podcasts, interviews, and talk shows—automatically extracting
            30–60s highlights of key <b>Q&A</b>, <b>Agreements</b>, and <b>Disagreements</b> snapped cleanly to camera transitions.
        </div>
        <div class="pill-row">
            <span class="tech-pill">Faster-Whisper ASR</span>
            <span class="tech-pill">WavLM + KMeans Diarization</span>
            <span class="tech-pill">DistilBERT-MNLI + YIN Pitch</span>
            <span class="tech-pill">PySceneDetect + OpenCV</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Controls
with st.sidebar:
    st.header("1. Video Source")
    input_mode = st.radio("Select Input Mode:", ["YouTube URL", "Upload MP4 File"])

    youtube_url = ""
    uploaded_file = None
    if input_mode == "YouTube URL":
        youtube_url = st.text_input(
            "Paste Interview URL:",
            placeholder="https://www.youtube.com/watch?v=...",
        )
    else:
        uploaded_file = st.file_uploader("Upload Interview MP4:", type=["mp4", "mov", "mkv"])

    st.divider()
    st.header("2. Extraction Parameters")
    buffer_window = st.slider("Context Buffer Window (s)", min_value=15, max_value=30, value=20)
    max_clips = st.slider("Max Clips to Generate", min_value=1, max_value=10, value=5)
    selected_categories = st.multiselect(
        "Highlight Categories",
        options=["QA", "Agreement", "Disagreement"],
        default=["QA", "Agreement", "Disagreement"],
    )

    st.divider()
    run_btn = st.button("Generate Highlights", type="primary", use_container_width=True)

# Main Execution Area
if not run_btn:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("**1. Speech & Pitch**\n\nFaster-Whisper transcribes dialogue with VAD filtering, while YIN F0 tracks rising intonation at question boundaries.")
    with c2:
        st.info("**2. Speaker Clustering**\n\nWavLM extracts voice embeddings and KMeans separates Host vs. Guest turns automatically.")
    with c3:
        st.info("**3. Semantic NLI**\n\nDistilBERT-MNLI evaluates cross-speaker responses to detect high-engagement Q&A, Agreements, and Debates.")
    with c4:
        st.info("**4. Vision Snapping**\n\nPySceneDetect aligns cuts to camera switches and OpenCV verifies active speaker faces within 30–60s windows.")

    st.caption("👈 Paste a YouTube interview URL or upload an MP4 file in the sidebar and click **Generate Highlights** to begin.")
else:
    if input_mode == "YouTube URL" and not youtube_url.strip():
        st.warning("Please enter a valid YouTube URL in the sidebar.")
        st.stop()
    if input_mode == "Upload MP4 File" and uploaded_file is None:
        st.warning("Please upload an MP4 video file in the sidebar.")
        st.stop()
    if not selected_categories:
        st.warning("Please select at least one highlight category.")
        st.stop()

    base_tmp = Path(tempfile.mkdtemp(prefix="echoframe_"))
    work_dir = base_tmp / "work"
    output_dir = base_tmp / "outputs"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with st.status("Running Multimodal Highlight Pipeline...", expanded=True) as status:
        st.write("Step 1/5: Preparing video & extracting 16kHz audio...")
        if input_mode == "YouTube URL":
            video_path = download_youtube_video(youtube_url.strip(), work_dir)
        else:
            video_path = str(work_dir / "uploaded_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

        audio_path = extract_audio(video_path, work_dir)
        total_duration = get_video_duration(video_path)

        st.write("Step 2/5: Transcribing speech & analyzing pitch intonation (Whisper-tiny)...")
        df_sentences = transcribe_and_merge(audio_path)
        df_sentences = detect_pitch_questions(df_sentences, audio_path)

        st.write("Step 3/5: Separating speakers (WavLM + KMeans)...")
        df_sentences = assign_speakers(df_sentences, audio_path)

        st.write("Step 4/5: Classifying Q&A, Agreements & Debates (DistilBERT-MNLI)...")
        df_moments = detect_highlight_moments(df_sentences, selected_categories)

        if df_moments.empty:
            status.update(label="Analysis complete — no matching highlights found.", state="complete")
            st.info("No segments exceeded the confidence threshold for the selected categories. Try selecting all categories or testing a longer dialogue video.")
            st.stop()

        st.write("Step 5/5: Snapping camera cuts, scoring faces & rendering MP4 clips...")
        df_enriched = enrich_with_vision(df_moments, video_path, total_duration, buffer_y=float(buffer_window))
        df_final = filter_non_overlapping(df_enriched, max_clips=max_clips)
        df_final = extract_highlight_clips(video_path, df_final, output_dir)

        status.update(label=f"Pipeline Complete! Generated {len(df_final)} highlight clips.", state="complete")

    # Summary Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Video Duration", f"{round(total_duration / 60, 1)} min")
    m2.metric("Sentences Analyzed", len(df_sentences))
    m3.metric("Highlights Extracted", len(df_final))

    tab_clips, tab_transcript = st.tabs(["🎬 Generated Highlight Clips", "📝 Full Speaker Transcript"])

    with tab_clips:
        for idx, row in df_final.iterrows():
            cat = row["category"]
            badge_cls = "badge-qa" if cat == "QA" else ("badge-agree" if cat == "Agreement" else "badge-disagree")

            with st.container(border=True):
                col_vid, col_meta = st.columns([1.2, 1.8])
                with col_vid:
                    clip_file = row.get("clip_path", "")
                    if clip_file and Path(clip_file).exists():
                        st.video(clip_file)
                        with open(clip_file, "rb") as vf:
                            st.download_button(
                                label=f"⬇️ Download Clip #{idx + 1} (.mp4)",
                                data=vf.read(),
                                file_name=f"highlight_{idx + 1}_{cat.lower()}.mp4",
                                mime="video/mp4",
                                key=f"dl_{idx}",
                            )
                with col_meta:
                    st.markdown(
                        f"#### Highlight #{idx + 1} &nbsp; <span class='{badge_cls}'>{cat}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"**Window:** {row['clip_start']}s – {row['clip_end']}s ({row['clip_duration']}s) &nbsp;|&nbsp; "
                        f"**Speakers:** {row['speakers']} &nbsp;|&nbsp; "
                        f"**Score:** {row['final_multimodal_score']}"
                    )
                    st.markdown(f"**Trigger:** *\"{row['trigger_text']}\"*")
                    st.markdown(f"**Response:** *\"{row['response_text']}\"*")

    with tab_transcript:
        st.dataframe(
            df_sentences[["start", "end", "speaker", "is_question", "text"]],
            use_container_width=True,
            hide_index=True,
        )