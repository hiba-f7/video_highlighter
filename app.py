import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

from pipeline.downloader import download_youtube_video, extract_audio, get_video_duration
from pipeline.audio_nlp import transcribe_and_merge, detect_pitch_questions, detect_highlight_moments
from pipeline.diarization import assign_speakers
from pipeline.vision import enrich_with_vision
from pipeline.cutter import filter_non_overlapping, extract_highlight_clips

st.set_page_config(
    page_title="EchoFrame | Multimodal Interview Highlight Extractor",
    page_icon="🎬",
    layout="wide",
)

# Editorial Magazine Palette (#2E5968, #97B5C2, #F7E0E0, #DFC2C0, #C8B9B1, #18313A)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    .stApp {
        background: #FBF7F6 !important;
        color: #18313A !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Hide default Streamlit sidebar toggle & chrome */
    section[data-testid="stSidebar"] {
        display: none !important;
    }

    /* Editorial Masthead Ribbon */
    .masthead {
        background: #F7E0E0;
        border-bottom: 2px solid #2E5968;
        border-radius: 10px 10px 0 0;
        padding: 18px 24px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .masthead-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #18313A !important;
        margin: 0;
    }

    /* Section Headings */
    .stage-heading {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #18313A !important;
        margin-bottom: 14px;
    }
    .editorial-category {
        font-family: 'Playfair Display', serif;
        font-size: 1.55rem;
        font-weight: 700;
        color: #18313A !important;
        border-bottom: 1.5px solid #C8B9B1;
        padding-bottom: 6px;
        margin-top: 18px;
        margin-bottom: 12px;
    }

    /* Editorial Index Cards (Mist Blue & Soft Ivory) */
    .index-card-blue {
        background: rgba(151, 181, 194, 0.38);
        border: 1px solid #97B5C2;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .index-card-ivory {
        background: #FFFFFF;
        border: 1px solid #DFC2C0;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .index-time {
        font-size: 0.82rem;
        font-weight: 700;
        color: #18313A !important;
        margin-bottom: 4px;
    }
    .index-quote {
        font-size: 0.84rem;
        line-height: 1.45;
        color: #18313A !important;
        margin-bottom: 4px;
    }

    /* Primary Button: High-Contrast Light Text (#FFFFFF / #FBF7F6) on Deep Slate Teal */
    div.stButton > button[kind="primary"] {
        background: #2E5968 !important;
        border: 1px solid #2E5968 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
    }
    div.stButton > button[kind="primary"] *,
    div.stButton > button[kind="primary"] p,
    div.stButton > button[kind="primary"] span {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #18313A !important;
        border-color: #18313A !important;
    }

    /* Secondary Pill Buttons */
    div.stButton > button[kind="secondary"] {
        background: #FBF7F6 !important;
        border: 1.5px solid #2E5968 !important;
        color: #18313A !important;
        font-weight: 600 !important;
        border-radius: 999px !important;
        font-size: 0.78rem !important;
        padding: 4px 14px !important;
    }
    div.stButton > button[kind="secondary"] *,
    div.stButton > button[kind="secondary"] p,
    div.stButton > button[kind="secondary"] span {
        color: #18313A !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: #2E5968 !important;
        color: #FFFFFF !important;
    }
    div.stButton > button[kind="secondary"]:hover *,
    div.stButton > button[kind="secondary"]:hover p,
    div.stButton > button[kind="secondary"]:hover span {
        color: #FFFFFF !important;
    }

    .stTextInput input, .stFileUploader section {
        background-color: #FFFFFF !important;
        color: #18313A !important;
        border: 1px solid #C8B9B1 !important;
        border-radius: 8px !important;
    }
    label, .stRadio span, .stSlider span {
        color: #18313A !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top Masthead (Clean Multimodal Title, No Right Subtitle)
st.markdown(
    """
    <div class="masthead">
        <div class="masthead-title">ECHOFRAME: MULTIMODAL INTERVIEW HIGHLIGHT EXTRACTOR</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Horizontal Control Bar
c_mode, c_input, c_max, c_action = st.columns([1.3, 3.2, 1.3, 1.3])

with c_mode:
    input_mode = st.radio("Source Mode", ["YouTube URL", "Upload MP4 Clip"], horizontal=True)

with c_input:
    youtube_url = ""
    uploaded_file = None
    if input_mode == "YouTube URL":
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://youtube.com/watch?v=...",
            label_visibility="collapsed",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload MP4",
            type=["mp4", "mov", "mkv"],
            label_visibility="collapsed",
        )

with c_max:
    max_per_category = st.slider("Clips per Group", min_value=1, max_value=4, value=2)

with c_action:
    st.write("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    run_btn = st.button("Extract Highlights", type="primary", use_container_width=True)

st.write("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Session State Initialization
if "selected_clip_path" not in st.session_state:
    st.session_state["selected_clip_path"] = None
if "selected_clip_label" not in st.session_state:
    st.session_state["selected_clip_label"] = ""
if "selected_clip_quotes" not in st.session_state:
    st.session_state["selected_clip_quotes"] = ("", "")


def select_highlight(path: str, label: str, trig: str, resp: str):
    st.session_state["selected_clip_path"] = path
    st.session_state["selected_clip_label"] = label
    st.session_state["selected_clip_quotes"] = (trig, resp)


# Run Pipeline When Triggered
if run_btn:
    if input_mode == "YouTube URL" and not youtube_url.strip():
        st.warning("Please paste a YouTube URL in the control bar above.")
        st.stop()
    if input_mode == "Upload MP4 Clip" and uploaded_file is None:
        st.warning("Please upload an MP4 video file in the control bar above.")
        st.stop()

    base_tmp = Path(tempfile.mkdtemp(prefix="echoframe_ed_"))
    work_dir = base_tmp / "work"
    output_dir = base_tmp / "outputs"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with st.spinner("Running Multimodal Highlight Pipeline (Whisper-tiny + WavLM + DistilBERT-MNLI)..."):
        if input_mode == "YouTube URL":
            video_path = download_youtube_video(youtube_url.strip(), work_dir)
        else:
            video_path = str(work_dir / "uploaded_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

        audio_path = extract_audio(video_path, work_dir)
        total_duration = get_video_duration(video_path)

        df_sentences = transcribe_and_merge(audio_path)
        df_sentences = detect_pitch_questions(df_sentences, audio_path)
        df_sentences = assign_speakers(df_sentences, audio_path)
        df_moments = detect_highlight_moments(df_sentences, ["QA", "Agreement", "Disagreement"])

        sections = {"QA": pd.DataFrame(), "Agreement": pd.DataFrame(), "Disagreement": pd.DataFrame()}
        all_clips = []

        if not df_moments.empty:
            df_enriched = enrich_with_vision(df_moments, video_path, total_duration, buffer_y=0.8)
            for cat in ["QA", "Agreement", "Disagreement"]:
                sub = df_enriched[df_enriched["category"] == cat].copy()
                sub = filter_non_overlapping(sub, max_clips=int(max_per_category))
                sub = extract_highlight_clips(video_path, sub, output_dir / cat.lower())
                sections[cat] = sub
                for idx, r in sub.iterrows():
                    all_clips.append(
                        {
                            "pill": f"PLAY {cat.upper()} #{idx + 1}",
                            "label": f"{cat.upper()} #{idx + 1} ({r['clip_start']}s – {r['clip_end']}s | {r['speakers']})",
                            "path": r["clip_path"],
                            "trig": r["trigger_text"],
                            "resp": r["response_text"],
                        }
                    )

        st.session_state["sections"] = sections
        st.session_state["all_clips"] = all_clips
        st.session_state["df_sentences"] = df_sentences
        st.session_state["df_loaded"] = True

        if all_clips:
            first = all_clips[0]
            select_highlight(first["path"], first["label"], first["trig"], first["resp"])
        else:
            select_highlight(None, "", "", "")

# Main Editorial Spread: Left Cinema Stage | Right Editorial Index
col_stage, col_index = st.columns([1.45, 1.25], gap="large")

with col_stage:
    st.markdown('<div class="stage-heading">CINEMA SCREENING STAGE</div>', unsafe_allow_html=True)

    curr_path = st.session_state.get("selected_clip_path")
    curr_label = st.session_state.get("selected_clip_label", "")
    curr_trig, curr_resp = st.session_state.get("selected_clip_quotes", ("", ""))

    if curr_path and Path(curr_path).exists():
        st.video(curr_path)
        st.markdown(
            f"<div style='font-family:\"Playfair Display\", serif; font-size:1.15rem; font-weight:700; color:#2E5968; margin:10px 0 6px 0;'>Now Screening: {curr_label}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="index-card-ivory" style="border-left: 4px solid #2E5968;">
                <div class="index-quote"><b>Trigger:</b> "{curr_trig}"</div>
                <div class="index-quote"><b>Response:</b> "{curr_resp}"</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with open(curr_path, "rb") as vf:
            st.download_button(
                label="⬇️ Download Selected Highlight (.mp4)",
                data=vf.read(),
                file_name="echoframe_highlight.mp4",
                mime="video/mp4",
            )
    else:
        st.markdown(
            """
            <div style="background:#18313A; border:2px solid #2E5968; border-radius:14px; height:340px; display:flex; flex-direction:column; justify-content:center; align-items:center; padding:24px; text-align:center;">
                <div style="font-size:3.2rem; margin-bottom:10px;">🎬</div>
                <div style="font-family:'Playfair Display', serif; font-size:1.25rem; color:#F7E0E0 !important; font-weight:600;">
                    Cinema Screening Stage Ready
                </div>
                <div style="font-size:0.86rem; color:#97B5C2 !important; margin-top:6px;">
                    Paste a YouTube link or upload an MP4 above and click Extract Highlights to load 30s–60s clips.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Pill Selector Buttons Beneath the Video Player
    all_clips = st.session_state.get("all_clips", [])
    if all_clips:
        st.write("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        pill_cols = st.columns(min(len(all_clips), 3))
        for i, clip_item in enumerate(all_clips):
            with pill_cols[i % len(pill_cols)]:
                if st.button(
                    clip_item["pill"],
                    key=f"pill_btn_{i}",
                    type="secondary",
                    use_container_width=True,
                ):
                    select_highlight(
                        clip_item["path"],
                        clip_item["label"],
                        clip_item["trig"],
                        clip_item["resp"],
                    )
                    st.rerun()

    if "df_sentences" in st.session_state and not st.session_state["df_sentences"].empty:
        st.write("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        with st.expander("📝 View Full Speaker-Diarized Transcript"):
            st.dataframe(
                st.session_state["df_sentences"][["start", "end", "speaker", "is_question", "text"]],
                use_container_width=True,
                hide_index=True,
            )

with col_index:
    st.markdown('<div class="stage-heading">EDITORIAL HIGHLIGHT INDEX</div>', unsafe_allow_html=True)

    sections = st.session_state.get("sections", {})

    for cat_key, cat_title in [
        ("QA", "01 / QUESTION & ANSWER"),
        ("Agreement", "02 / AGREEMENT"),
        ("Disagreement", "03 / DISAGREEMENT"),
    ]:
        st.markdown(f'<div class="editorial-category">{cat_title}</div>', unsafe_allow_html=True)
        df_cat = sections.get(cat_key, pd.DataFrame())

        if df_cat.empty:
            st.markdown(
                "<div style='font-size:0.84rem; color:#5C7680; margin-bottom:14px; font-style:italic;'>No verified exchanges in this category.</div>",
                unsafe_allow_html=True,
            )
        else:
            card_cols = st.columns(min(len(df_cat), 2))
            for idx, row in df_cat.iterrows():
                card_style = "index-card-blue" if idx % 2 == 0 else "index-card-ivory"
                label_str = f"{cat_key.upper()} #{idx + 1} ({row['clip_start']}s – {row['clip_end']}s | {row['speakers']})"
                short_resp = (
                    row["response_text"][:135] + "..."
                    if len(row["response_text"]) > 135
                    else row["response_text"]
                )
                with card_cols[idx % len(card_cols)]:
                    st.markdown(
                        f"""
                        <div class="{card_style}">
                            <div class="index-time">
                                {row['clip_start']}s – {row['clip_end']}s ({row['clip_duration']}s) &bull; {row['speakers']}
                            </div>
                            <div class="index-quote"><b>Q/Trigger:</b> "{row['trigger_text'][:95]}"</div>
                            <div class="index-quote"><b>Response:</b> "{short_resp}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        f"▶ Screen {cat_key} #{idx + 1}",
                        key=f"idx_btn_{cat_key}_{idx}",
                        type="secondary",
                        use_container_width=True,
                    ):
                        select_highlight(
                            row["clip_path"],
                            label_str,
                            row["trigger_text"],
                            row["response_text"],
                        )
                        st.rerun()
