import tempfile
from pathlib import Path
import gradio as gr

from pipeline.downloader import download_youtube_video, extract_audio, get_video_duration
from pipeline.audio_nlp import transcribe_and_merge, detect_pitch_questions, detect_highlight_moments
from pipeline.diarization import assign_speakers
from pipeline.vision import enrich_with_vision
from pipeline.cutter import filter_non_overlapping, extract_highlight_clips

CUSTOM_CSS = """
.gradio-container {
    background: linear-gradient(160deg, #FBF7F6 0%, #F7E0E0 48%, #EEF4F6 100%) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
.echo-hero {
    background: linear-gradient(135deg, #2E5968 0%, #3E6E7E 60%, #97B5C2 100%);
    border: 1px solid #C8B9B1;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(46, 89, 104, 0.14);
}
.echo-hero h1 {
    color: #FBF7F6 !important;
    font-size: 1.85rem !important;
    margin: 0 0 6px 0 !important;
}
.echo-hero p {
    color: #F7E0E0 !important;
    font-size: 0.96rem !important;
    margin: 0 0 12px 0 !important;
}
.echo-pill {
    display: inline-block;
    background: rgba(247, 224, 224, 0.22);
    border: 1px solid #F7E0E0;
    color: #FFFFFF;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
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
    color: #2E5968;
    font-weight: 700;
    font-size: 0.92rem;
    margin-bottom: 6px;
}
.clip-quote {
    color: #334155;
    font-size: 0.9rem;
    margin-bottom: 4px;
}
button.primary {
    background: #2E5968 !important;
    border-color: #2E5968 !important;
    color: #FFFFFF !important;
}
button.primary:hover {
    background: #234652 !important;
}
"""


def build_section_outputs(df_cat, title):
    if df_cat.empty:
        empty_html = (
            f"<div class='clip-card' style='border-left-color:#C8B9B1;'>"
            f"<div class='clip-meta'>No high-confidence {title} exchanges found</div>"
            f"<div class='clip-quote'>Only verified cross-speaker exchanges passing the 0.72 confidence threshold are shown.</div>"
            f"</div>"
        )
        return empty_html, None, gr.update(choices=[], value=None, visible=False), []

    cards = []
    choices = []
    paths = []
    for idx, row in df_cat.iterrows():
        label = f"Clip #{idx + 1} ({row['clip_start']}s – {row['clip_end']}s | {row['speakers']})"
        choices.append(label)
        paths.append(row["clip_path"])
        cards.append(
            f"<div class='clip-card'>"
            f"<div class='clip-meta'>{label} &nbsp;•&nbsp; Confidence: {row['final_multimodal_score']}</div>"
            f"<div class='clip-quote'><b>Trigger:</b> \"{row['trigger_text']}\"</div>"
            f"<div class='clip-quote'><b>Response:</b> \"{row['response_text']}\"</div>"
            f"</div>"
        )

    return (
        "".join(cards),
        paths[0],
        gr.update(choices=choices, value=choices[0], visible=len(choices) > 1),
        paths,
    )


def switch_clip(selected_label, choices_list, paths_list):
    if not selected_label or not paths_list:
        return None
    try:
        idx = choices_list.index(selected_label)
        return paths_list[idx]
    except ValueError:
        return paths_list[0]


def run_pipeline(youtube_url, uploaded_video, max_per_category, progress=gr.Progress()):
    if not youtube_url and not uploaded_video:
        raise gr.Error("Please paste a YouTube URL or upload an MP4 video file.")

    base_tmp = Path(tempfile.mkdtemp(prefix="echoframe_gr_"))
    work_dir = base_tmp / "work"
    output_dir = base_tmp / "outputs"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress(0.1, desc="Step 1/5: Preparing video & extracting 16kHz audio...")
    if youtube_url and youtube_url.strip():
        video_path = download_youtube_video(youtube_url.strip(), work_dir)
    else:
        video_path = uploaded_video

    audio_path = extract_audio(video_path, work_dir)
    total_duration = get_video_duration(video_path)

    progress(0.3, desc="Step 2/5: Transcribing speech & detecting questions (Whisper-tiny + YIN Pitch)...")
    df_sentences = transcribe_and_merge(audio_path)
    df_sentences = detect_pitch_questions(df_sentences, audio_path)

    progress(0.55, desc="Step 3/5: Identifying speaker turns (WavLM + KMeans)...")
    df_sentences = assign_speakers(df_sentences, audio_path)

    progress(0.75, desc="Step 4/5: Filtering high-confidence QA, Agreement & Disagreement turns...")
    df_moments = detect_highlight_moments(df_sentences, ["QA", "Agreement", "Disagreement"])

    if df_moments.empty:
        empty_html = "<div class='clip-card'><div class='clip-meta'>No high-confidence cross-speaker exchanges detected.</div></div>"
        hide_dd = gr.update(choices=[], value=None, visible=False)
        return (
            empty_html, None, hide_dd, [],
            empty_html, None, hide_dd, [],
            empty_html, None, hide_dd, [],
            df_sentences[["start", "end", "speaker", "is_question", "text"]] if not df_sentences.empty else df_sentences,
        )

    progress(0.9, desc="Step 5/5: Snapping camera cuts & cutting exact exchange clips...")
    df_enriched = enrich_with_vision(df_moments, video_path, total_duration, buffer_y=1.5)

    sections = {}
    for cat in ["QA", "Agreement", "Disagreement"]:
        sub = df_enriched[df_enriched["category"] == cat].copy()
        sub = filter_non_overlapping(sub, max_clips=int(max_per_category))
        sub = extract_highlight_clips(video_path, sub, output_dir / cat.lower())
        sections[cat] = sub

    qa_html, qa_vid, qa_dd, qa_paths = build_section_outputs(sections["QA"], "Question & Answer")
    ag_html, ag_vid, ag_dd, ag_paths = build_section_outputs(sections["Agreement"], "Agreement")
    dis_html, dis_vid, dis_dd, dis_paths = build_section_outputs(sections["Disagreement"], "Disagreement")

    transcript_df = df_sentences[["start", "end", "speaker", "is_question", "text"]]
    return (
        qa_html, qa_vid, qa_dd, qa_paths,
        ag_html, ag_vid, ag_dd, ag_paths,
        dis_html, dis_vid, dis_dd, dis_paths,
        transcript_df,
    )


theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#EEF4F6", c100="#DCE8ED", c200="#B9D1DC", c300="#97B5C2",
        c400="#628C9D", c500="#2E5968", c600="#274C59", c700="#203F4A",
        c800="#18313A", c900="#11232A", c950="#0B171C",
    ),
    secondary_hue=gr.themes.Color(
        c50="#FDF8F8", c100="#F7E0E0", c200="#EBD1D0", c300="#DFC2C0",
        c400="#D4BDB6", c500="#C8B9B1", c600="#A89890", c700="#877871",
        c800="#665A54", c900="#453C38", c950="#2B2523",
    ),
)

with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="EchoFrame | Multimodal Interview Analyzer") as demo:
    gr.HTML(
        """
        <div class="echo-hero">
            <h1>EchoFrame: Multimodal Interview Highlight Extractor</h1>
            <p>Precision extraction of <b>Question &amp; Answer</b>, <b>Agreement</b>, and <b>Disagreement</b> exchanges—tightly bounded to individual speaker turns.</p>
            <span class="echo-pill">Whisper-tiny ASR + YIN Pitch</span>
            <span class="echo-pill">WavLM + KMeans Diarization</span>
            <span class="echo-pill">DistilBERT-MNLI Semantic Gate</span>
            <span class="echo-pill">PySceneDetect + OpenCV</span>
        </div>
        """
    )

    qa_paths_state = gr.State([])
    ag_paths_state = gr.State([])
    dis_paths_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            yt_input = gr.Textbox(label="YouTube Interview URL", placeholder="https://www.youtube.com/watch?v=...")
            vid_input = gr.Video(label="Or Upload MP4 Clip")
            max_slider = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Max Clips per Category")
            run_btn = gr.Button("Extract Sectioned Highlights", variant="primary")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Question & Answer"):
                    qa_selector = gr.Dropdown(label="Select Q&A Clip", visible=False)
                    qa_video = gr.Video(label="Question & Answer Clip")
                    qa_info = gr.HTML()

                with gr.Tab("Agreement"):
                    ag_selector = gr.Dropdown(label="Select Agreement Clip", visible=False)
                    ag_video = gr.Video(label="Agreement Clip")
                    ag_info = gr.HTML()

                with gr.Tab("Disagreement"):
                    dis_selector = gr.Dropdown(label="Select Disagreement Clip", visible=False)
                    dis_video = gr.Video(label="Disagreement Clip")
                    dis_info = gr.HTML()

                with gr.Tab("Speaker Transcript"):
                    transcript_out = gr.Dataframe(interactive=False)

    run_btn.click(
        fn=run_pipeline,
        inputs=[yt_input, vid_input, max_slider],
        outputs=[
            qa_info, qa_video, qa_selector, qa_paths_state,
            ag_info, ag_video, ag_selector, ag_paths_state,
            dis_info, dis_video, dis_selector, dis_paths_state,
            transcript_out,
        ],
    )

    def _pick_clip(selected_label, paths):
        if not selected_label or not paths:
            return None
        try:
            idx = int(selected_label.split("#")[1].split(" ")[0]) - 1
            return paths[idx]
        except Exception:
            return paths[0]

    qa_selector.change(fn=_pick_clip, inputs=[qa_selector, qa_paths_state], outputs=qa_video)
    ag_selector.change(fn=_pick_clip, inputs=[ag_selector, ag_paths_state], outputs=ag_video)
    dis_selector.change(fn=_pick_clip, inputs=[dis_selector, dis_paths_state], outputs=dis_video)

if __name__ == "__main__":
    demo.launch(share=True, debug=False)
