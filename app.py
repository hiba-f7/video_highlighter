import tempfile
from pathlib import Path
import streamlit as st

from pipeline.downloader import download_youtube_video, extract_audio, get_video_duration
from pipeline.audio_nlp import transcribe_and_merge, detect_pitch_questions, detect_highlight_moments
from pipeline.diarization import assign_speakers
from pipeline.vision import enrich_with_vision
from pipeline.cutter import filter_non_overlapping, extract_highlight_clips

st.set_page_config(
    page_title="EchoFrame | Multimodal Interview Highlight Extractor",
    page_icon="🔍",
    layout="wide",
)

# Custom Palette Styling (#2E5968, #97B5C2, #F7E0E0, #DFC2C0, #C8B9B1)
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(160deg, #FBF7F6 0%, #F7E0E0 50%, #EEF4F6 100%) !important;
        color: #18313A !important;
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #DFC2C0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #18313A !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown span, label, h1, h2, h3, h4 {
        color: #18313A !important;
    }
    .stTextInput input, .stFileUploader section {
        background-color: #FBF7F6 !important;
        color: #18313A !important;
        border: 1px solid #97B5C2 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255, 255, 255, 0.75);
        border-radius: 10px;
        padding: 4px;
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #2E5968 !important;
        font-weight: 700 !important;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #18313A !important;
        border-bottom: 3px solid #2E5968 !important;
    }
    .echo-hero {
        background: linear-gradient(135deg, #2E5968 0%, #3E6E7E 65%, #97B5C2 100%);
        border: 1px solid #C8B9B1;
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 18px;
        box-shadow: 0 8px 22px rgba(46, 89, 104, 0.14);
    }
    .echo-hero h1 {
        color: #FFFFFF !important;
        font-size: 1.8rem !important;
        margin: 0 0 6px 0 !important;
    }
    .echo-hero p {
        color: #F7E0E0 !important;
        font-size: 0.96rem !important;
        margin: 0 0 12px 0 !important;
    }
    .echo-pill {
        display: inline-block;
        background: rgba(247, 224, 224, 0.24);
        border: 1px solid #F7E0E0;
        color: #FFFFFF !important;
        padding: 3px 11px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .clip-card {
        background: #FFFFFF;
        border: 1px solid #DFC2C0;
        border-left: 5px solid #2E5968;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(46, 89, 104, 0.06);
    }
    .clip-meta {
        color: #2E5968 !important;
        font-weight: 700;
        font-size: 0.94rem;
        margin-bottom: 6px;
    }
    .clip-quote {
        color: #18313A !important;
        font-size: 0.91rem;
        margin-bottom: 4px;
    }
    div.stButton > button[kind="primary"] {
        background-color: #2E5968 !important;
        border-color: #2E5968 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #18313A !important;
        border-color: #18313A !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero Banner
st.markdown(
    """
    <div class="echo-hero">
        <h1>EchoFrame: Multimodal Interview Highlight Extractor</h1>
        <p>Precision extraction of <b>Question &amp; Answer</b>, <b>Agreement</b>, and <b>Disagreement</b> exchanges—cut strictly at complete sentence boundaries.</p>
        <span class="echo-pill">Whisper-tiny ASR + YIN Pitch</span>
        <span class="echo-pill">WavLM + KMeans Diarization</span>
        <span class="echo-pill">DistilBERT-MNLI (0.50 Gate)</span>
        <span class="echo-pill">OpenCV Face Verification</span>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_category_tab(df_cat, title: str, tab_key: str):
    """Renders verified clips and transcripts for a single category tab."""
    if df_cat.empty:
        st.markdown(
            f"""
            <div class="clip-card" style="border-left-color:#C8B9B1;">
                <div class="clip-meta">No {title} exchanges detected in this segment</div>
                <div class="clip-quote">Threshold is set to 0.50—if no {title.lower()} occurred between speakers, this tab stays clean.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for idx, row in df_cat.iterrows():
        with st.container():
            col_vid, col_info = st.columns([1.15, 1.85])
            with col_vid:
                clip_file = row.get("clip_path", "")
                if clip_file and Path(clip_file).exists():
                    st.video(clip_file)
                    with open(clip_file, "rb") as vf:
                        st.download_button(
                            label=f"⬇️ Download {title} Clip #{idx + 1} (.mp4)",
                            data=vf.read(),
                            file_name=f"{tab_key}_clip_{idx + 1}.mp4",
                            mime="video/mp4",
                            key=f"dl_{tab_key}_{idx}",
                        )
            with col_info:
                st.markdown(
                    f"""
                    <div class="clip-card">
                        <div class="clip-meta">
                            Clip #{idx + 1} ({row['clip_start']}s – {row['clip_end']}s | {row['speakers']})
                            &nbsp;•&nbsp; Confidence: {row['final_multimodal_score']}
                        </div>
                        <div class="clip-quote"><b>Trigger:</b> "{row['trigger_text']}"</div>
                        <div class="clip-quote"><b>Response:</b> "{row['response_text']}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# Sidebar Controls
with st.sidebar:
    st.header("1. Video Source")
    input_mode = st.radio("Select Input Mode:", ["YouTube URL", "Upload MP4 Clip"])

    youtube_url = ""
    uploaded_file = None
    if input_mode == "YouTube URL":
        youtube_url = st.text_input(
            "YouTube Interview URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )
    else:
        uploaded_file = st.file_uploader("Upload MP4 Clip", type=["mp4", "mov", "mkv"])

    st.divider()
    st.header("2. Extraction Settings")
    max_per_category = st.slider("Max Clips per Category", min_value=1, max_value=4, value=2)

    st.divider()
    run_btn = st.button("Extract Sectioned Highlights", type="primary", use_container_width=True)

# Landing Overview
if not run_btn:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="clip-card">
                <div class="clip-meta">1. Speech &amp; Pitch</div>
                <div class="clip-quote">Whisper-tiny transcribes complete sentences while YIN F0 tracks rising question intonation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="clip-card">
                <div class="clip-meta">2. Speaker Clustering</div>
                <div class="clip-quote">WavLM extracts voice embeddings and KMeans separates Interviewer vs. Guest turns.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="clip-card">
                <div class="clip-meta">3. Semantic Gate (0.50)</div>
                <div class="clip-quote">DistilBERT-MNLI separates Q&amp;A, Agreements, and Disagreements into dedicated sections.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="clip-card">
                <div class="clip-meta">4. Complete-Sentence Cuts</div>
                <div class="clip-quote">OpenCV verifies active speaker faces and FFmpeg cuts cleanly after full sentences finish.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.info("👈 Paste a YouTube interview URL or upload an MP4 clip in the sidebar, then click **Extract Sectioned Highlights**.")
else:
    if input_mode == "YouTube URL" and not youtube_url.strip():
        st.warning("Please paste a valid YouTube URL in the sidebar.")
        st.stop()
    if input_mode == "Upload MP4 Clip" and uploaded_file is None:
        st.warning("Please upload an MP4 video clip in the sidebar.")
        st.stop()

    base_tmp = Path(tempfile.mkdtemp(prefix="echoframe_st_"))
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

        st.write("Step 2/5: Transcribing complete sentences (Whisper-tiny + YIN Pitch)...")
        df_sentences = transcribe_and_merge(audio_path)
        df_sentences = detect_pitch_questions(df_sentences, audio_path)

        st.write("Step 3/5: Separating speakers (WavLM + KMeans)...")
        df_sentences = assign_speakers(df_sentences, audio_path)

        st.write("Step 4/5: Classifying Q&A, Agreement & Disagreement (Threshold 0.50)...")
        df_moments = detect_highlight_moments(df_sentences, ["QA", "Agreement", "Disagreement"])

        if df_moments.empty:
            status.update(label="Analysis complete — no exchanges passed the 0.50 confidence threshold.", state="complete")
            st.warning("No cross-speaker exchanges exceeded the 0.50 confidence threshold in this clip.")
            st.stop()

        st.write("Step 5/5: Verifying speaker faces & cutting complete-sentence clips...")
        df_enriched = enrich_with_vision(df_moments, video_path, total_duration, buffer_y=1.2)

        sections = {}
        for cat in ["QA", "Agreement", "Disagreement"]:
            sub = df_enriched[df_enriched["category"] == cat].copy()
            sub = filter_non_overlapping(sub, max_clips=int(max_per_category))
            sub = extract_highlight_clips(video_path, sub, output_dir / cat.lower())
            sections[cat] = sub

        total_clips = sum(len(sections[c]) for c in sections)
        status.update(label=f"Pipeline Complete! Extracted {total_clips} sectioned clips.", state="complete")

    # Summary Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Video Duration", f"{round(total_duration / 60, 1)} min")
    m2.metric("Q&A Clips", len(sections["QA"]))
    m3.metric("Agreement Clips", len(sections["Agreement"]))
    m4.metric("Disagreement Clips", len(sections["Disagreement"]))

    tab_qa, tab_ag, tab_dis, tab_tr = st.tabs(
        ["❓ Question & Answer", "🤝 Agreement", "⚡ Disagreement", "📝 Speaker Transcript"]
    )

    with tab_qa:
        render_category_tab(sections["QA"], "Question & Answer", "qa")

    with tab_ag:
        render_category_tab(sections["Agreement"], "Agreement", "agreement")

    with tab_dis:
        render_category_tab(sections["Disagreement"], "Disagreement", "disagreement")

    with tab_tr:
        st.dataframe(
            df_sentences[["start", "end", "speaker", "is_question", "text"]],
            use_container_width=True,
            hide_index=True,
        )
