"""IndustrialEye AI Streamlit application."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from src.config import (
    CUSTOM_MODEL_NAME,
    MODELS_DIR,
    ProcessingConfig,
    VIDEO_EXTENSIONS,
    ensure_directories,
)
from src.detection.detector import YOLODetector
from src.detection.model_loader import (
    get_device_info,
    get_model_classes,
    load_yolo_model,
    model_path_for_mode,
    try_set_open_vocab_classes,
)
from src.processing.video_processor import LiveStats, VideoProcessor
from src.ui.components import (
    make_rate_chart,
    make_timeline_chart,
    render_future_features_note,
    render_header,
    render_kpi_cards,
    render_results,
    render_video_info,
)
from src.ui.styles import load_css
from src.utils.file_utils import cleanup_old_files, save_uploaded_file
from src.utils.video_utils import find_latest_downloads_video, get_video_info


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def cached_model(model_source: str):
    """Cache YOLO model loading across Streamlit reruns."""

    return load_yolo_model(Path(model_source))


def initialize_state() -> None:
    """Initialize durable Streamlit state."""

    defaults = {
        "uploaded_video_path": None,
        "video_info": None,
        "processing": False,
        "result": None,
        "events": [],
        "last_error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_sidebar() -> tuple[ProcessingConfig, str | None]:
    """Render configuration controls and return processing config."""

    st.sidebar.markdown("## DETECTION MODEL")
    model_mode = st.sidebar.radio("Model", ["Standard YOLO", "Open Vocabulary", "Custom Model"], horizontal=False)

    custom_path = None
    if model_mode == "Custom Model":
        default_custom = MODELS_DIR / CUSTOM_MODEL_NAME
        uploaded_model = st.sidebar.file_uploader("Upload .pt model", type=["pt"], key="custom_model_upload")
        if uploaded_model:
            custom_path = MODELS_DIR / f"custom_{uploaded_model.name}"
            custom_path.write_bytes(uploaded_model.getbuffer())
        elif default_custom.exists():
            custom_path = default_custom
            st.sidebar.caption(f"Using {default_custom.name}")
        else:
            st.sidebar.warning("Upload a .pt model or place manufacturing_best.pt in models/.")

    confidence = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.35, 0.05)
    iou = st.sidebar.slider("IoU Threshold", 0.10, 1.00, 0.50, 0.05)

    st.sidebar.markdown("## TARGET OBJECTS")
    open_vocab_text = ""
    selected_classes: list[str] = []
    model_warning = None
    model_source = model_path_for_mode(model_mode, custom_path)

    available_classes: list[str] = []
    if model_mode != "Open Vocabulary":
        try:
            preview_model = cached_model(str(model_source))
            available_classes = get_model_classes(preview_model)
        except Exception as exc:  # noqa: BLE001
            model_warning = f"Model classes could not be read yet: {exc}"
        selected_classes = st.sidebar.multiselect(
            "Classes to count",
            available_classes,
            default=[],
            help="Leave empty to count all detected classes.",
        )
    else:
        open_vocab_text = st.sidebar.text_input("Target labels", "box, package, carton, bottle, product, component")

    st.sidebar.markdown("## TRACKING")
    st.sidebar.selectbox("Tracker", ["ByteTrack"], index=0, disabled=True)

    st.sidebar.markdown("## COUNTING")
    counting_mode = st.sidebar.radio("Counting Mode", ["Horizontal Line", "Vertical Line", "ROI Zone"])
    line_position = st.sidebar.slider("Line / Zone Position", 0.05, 0.95, 0.50, 0.01)
    count_direction = st.sidebar.radio("Count Direction", ["Both", "Forward", "Reverse"], horizontal=True)

    st.sidebar.markdown("## PROCESSING")
    frame_skip = st.sidebar.slider("Frame Skip", 0, 10, 0)
    resolution = st.sidebar.select_slider("Input Resolution", options=[320, 416, 512, 640, 768, 960], value=640)
    show_tracking_ids = st.sidebar.checkbox("Show Tracking IDs", True)
    show_confidence = st.sidebar.checkbox("Show Confidence", True)
    show_trail = st.sidebar.checkbox("Show Trail", True)
    show_counting_line = st.sidebar.checkbox("Show Counting Line", True)
    show_fps = st.sidebar.checkbox("Show FPS", True)
    debug = st.sidebar.checkbox("Debug Mode", False)

    target_labels = [item.strip() for item in open_vocab_text.split(",") if item.strip()]
    config = ProcessingConfig(
        model_mode=model_mode,
        model_path=str(model_source),
        target_labels=target_labels,
        selected_classes=selected_classes,
        confidence_threshold=confidence,
        iou_threshold=iou,
        counting_mode=counting_mode,
        line_position=line_position,
        count_direction=count_direction,
        frame_skip=frame_skip,
        input_resolution=resolution,
        show_tracking_ids=show_tracking_ids,
        show_confidence=show_confidence,
        show_trail=show_trail,
        show_counting_line=show_counting_line,
        show_fps=show_fps,
        debug=debug,
    )
    return config, model_warning


def reset_state() -> None:
    """Reset analysis-related state without clearing cached models."""

    st.session_state.uploaded_video_path = None
    st.session_state.video_info = None
    st.session_state.result = None
    st.session_state.events = []
    st.session_state.last_error = None


def load_default_sample_video() -> None:
    """Use the newest Downloads video as a ready-to-run sample when available."""

    if st.session_state.uploaded_video_path or st.session_state.video_info:
        return
    sample_path = find_latest_downloads_video()
    if not sample_path:
        return
    try:
        st.session_state.uploaded_video_path = sample_path
        st.session_state.video_info = get_video_info(sample_path)
        st.session_state.last_error = None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Downloads sample video could not be loaded: %s", exc)


def main() -> None:
    """Run the Streamlit application."""

    st.set_page_config(page_title="IndustrialEye AI", page_icon="IE", layout="wide")
    ensure_directories()
    cleanup_old_files(Path("temp"))
    cleanup_old_files(Path("outputs"))
    initialize_state()
    load_default_sample_video()
    load_css()

    device_info = get_device_info()
    render_header(device_info.label)
    config, model_warning = render_sidebar()
    if model_warning:
        st.warning(model_warning)

    result = st.session_state.result
    if result:
        render_kpi_cards(
            result.summary.total_count,
            result.summary.average_throughput,
            result.summary.average_confidence,
            result.video_info.frame_count / max(result.summary.processing_time_seconds, 1e-6),
        )
    else:
        render_kpi_cards(0, 0.0, 0.0, 0.0)

    st.markdown('<div class="section-title">Upload Production Line Footage</div>', unsafe_allow_html=True)
    if st.session_state.video_info and st.session_state.uploaded_video_path:
        st.info("Sample video loaded from Downloads. You can click START AI ANALYSIS directly, or upload another video.")
    uploaded = st.file_uploader(
        "Supported: MP4, AVI, MOV, MKV",
        type=list(VIDEO_EXTENSIONS),
        accept_multiple_files=False,
        label_visibility="visible",
    )

    if uploaded and not st.session_state.processing:
        suffix = Path(uploaded.name).suffix
        try:
            path = save_uploaded_file(uploaded, suffix=suffix)
            info = get_video_info(path)
            st.session_state.uploaded_video_path = path
            st.session_state.video_info = info
            st.session_state.result = None
            st.session_state.last_error = None
        except Exception as exc:  # noqa: BLE001
            st.session_state.last_error = str(exc)
            if config.debug:
                st.exception(exc)
            else:
                st.error(str(exc))

    info = st.session_state.video_info
    if info:
        left, right = st.columns([1.25, 1])
        with left:
            st.video(str(info.path))
        with right:
            render_video_info(info)

    action_cols = st.columns([0.22, 0.18, 0.60])
    start = action_cols[0].button("START AI ANALYSIS", type="primary", disabled=not bool(st.session_state.uploaded_video_path))
    if action_cols[1].button("RESET"):
        reset_state()
        st.rerun()

    live_video = st.empty()
    live_kpis = st.empty()
    live_charts = st.empty()
    progress_bar = st.progress(0, text="Waiting for video analysis")

    if start and st.session_state.uploaded_video_path:
        st.session_state.processing = True
        try:
            with st.spinner("Loading AI model and starting ByteTrack analysis..."):
                model = cached_model(config.model_path)
                if config.model_mode == "Open Vocabulary":
                    ok, warning = try_set_open_vocab_classes(model, config.target_labels)
                    if not ok and warning:
                        st.warning(f"{warning} Falling back to the model's default vocabulary.")
                detector = YOLODetector(model=model, device=device_info.device)
                processor = VideoProcessor(detector=detector, config=config)

            def on_progress(stats: LiveStats) -> None:
                progress = min(stats.frame_number / max(stats.total_frames, 1), 1.0)
                progress_bar.progress(progress, text=f"Processing frame {stats.frame_number:,} of {stats.total_frames:,}")
                with live_video.container():
                    st.markdown('<div class="section-title">AI Processed Video</div>', unsafe_allow_html=True)
                    st.image(stats.annotated_frame, channels="RGB", use_container_width=True)
                with live_kpis.container():
                    render_kpi_cards(stats.total_count, stats.objects_per_minute, stats.average_confidence, stats.processing_fps)
                    cols = st.columns(3)
                    cols[0].metric("Forward", stats.forward_count)
                    cols[1].metric("Reverse", stats.reverse_count)
                    cols[2].metric("Events", len(stats.events))
                with live_charts.container():
                    if stats.events:
                        import pandas as pd

                        from src.analytics.metrics import build_timeline, events_to_dataframe

                        live_events = events_to_dataframe(stats.events)
                        timeline = build_timeline(live_events, stats.frame_number / max(info.fps, 1))
                        c1, c2 = st.columns(2)
                        c1.plotly_chart(make_timeline_chart(timeline), use_container_width=True)
                        c2.plotly_chart(make_rate_chart(timeline), use_container_width=True)

            result = processor.process(Path(st.session_state.uploaded_video_path), progress_callback=on_progress)
            st.session_state.result = result
            st.session_state.events = result.events_df.to_dict("records")
            progress_bar.progress(1.0, text="Analysis complete")
            if result.warning:
                st.warning(result.warning)
            st.rerun()
        except KeyboardInterrupt:
            st.warning("Processing was interrupted before completion.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Video analysis failed")
            st.session_state.last_error = str(exc)
            if config.debug:
                st.exception(exc)
            else:
                st.error(f"Analysis could not be completed: {exc}")
        finally:
            st.session_state.processing = False

    if st.session_state.result:
        render_results(
            summary=st.session_state.result.summary,
            events_df=st.session_state.result.events_df,
            timeline_df=st.session_state.result.timeline_df,
            class_distribution=st.session_state.result.class_distribution,
            confidence_values=st.session_state.result.confidence_values,
            output_video=st.session_state.result.output_video,
        )

    render_future_features_note()


if __name__ == "__main__":
    main()
