"""Reusable Streamlit UI components."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics.metrics import SummaryMetrics, summary_to_dataframe
from src.utils.video_utils import VideoInfo


def render_header(device_label: str) -> None:
    """Render the application header."""

    st.markdown(
        f"""
        <div class="hero">
            <div class="status-badge"><span class="status-dot"></span>AI Engine Ready</div>
            <h1>IndustrialEye AI</h1>
            <p>AI-powered Production Detection, Tracking &amp; Counting</p>
            <p style="margin-top:10px;color:#5ee7ff;font-weight:700;">AI Device: {device_label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(total: int, rate: float, avg_confidence: float, fps: float) -> None:
    """Render top KPI cards."""

    values = [
        ("TOTAL COUNT", f"{total:,}"),
        ("CURRENT RATE", f"{rate:.1f} objects/min"),
        ("AVG CONFIDENCE", f"{avg_confidence * 100:.1f}%"),
        ("PROCESSING FPS", f"{fps:.1f} FPS"),
    ]
    cols = st.columns(4)
    for col, (label, value) in zip(cols, values, strict=False):
        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-accent"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_video_info(info: VideoInfo) -> None:
    """Render uploaded video metadata."""

    st.markdown('<div class="section-title">Video Information</div>', unsafe_allow_html=True)
    data = {
        "File": info.path.name,
        "Resolution": f"{info.width} x {info.height}",
        "FPS": f"{info.fps:.2f}",
        "Duration": f"{info.duration_seconds:.1f} s",
        "Total Frames": f"{info.frame_count:,}",
        "File Size": f"{info.file_size_mb:.2f} MB",
    }
    st.dataframe(pd.DataFrame(data.items(), columns=["Field", "Value"]), hide_index=True, use_container_width=True)


def make_timeline_chart(timeline_df: pd.DataFrame):
    """Objects counted over time chart."""

    fig = px.line(timeline_df, x="second", y="cumulative_count", labels={"second": "Time (s)", "cumulative_count": "Objects"})
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
    return fig


def make_rate_chart(timeline_df: pd.DataFrame):
    """Production rate chart."""

    fig = px.area(timeline_df, x="second", y="objects_per_minute", labels={"second": "Time (s)", "objects_per_minute": "Objects/min"})
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
    return fig


def render_results(
    summary: SummaryMetrics,
    events_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    class_distribution: pd.DataFrame,
    confidence_values: list[float],
    output_video: Path,
) -> None:
    """Render the final analysis section."""

    st.success("ANALYSIS COMPLETE")
    cols = st.columns(4)
    cols[0].metric("Total Objects", f"{summary.total_count:,}")
    cols[1].metric("Forward Count", f"{summary.forward_count:,}")
    cols[2].metric("Reverse Count", f"{summary.reverse_count:,}")
    cols[3].metric("Average Confidence", f"{summary.average_confidence * 100:.1f}%")

    cols = st.columns(4)
    cols[0].metric("Average Throughput", f"{summary.average_throughput:.1f}/min")
    cols[1].metric("Peak Throughput", f"{summary.peak_throughput:.1f}/min")
    cols[2].metric("Processing Time", f"{summary.processing_time_seconds:.1f}s")
    spacing = "N/A" if summary.average_spacing_seconds is None else f"{summary.average_spacing_seconds:.2f}s"
    cols[3].metric("Avg Spacing", spacing)

    st.markdown('<div class="section-title">Processed Video Evidence</div>', unsafe_allow_html=True)
    st.video(str(output_video))

    left, right = st.columns(2)
    left.plotly_chart(make_timeline_chart(timeline_df), use_container_width=True, key="results_timeline")
    right.plotly_chart(make_rate_chart(timeline_df), use_container_width=True, key="results_rate")

    left, right = st.columns(2)
    if not class_distribution.empty:
        fig = px.bar(class_distribution, x="class", y="count", labels={"class": "Class", "count": "Count"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
        left.plotly_chart(fig, use_container_width=True, key="results_class_distribution")
    if confidence_values:
        fig = px.histogram(x=confidence_values, nbins=20, labels={"x": "Confidence"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
        right.plotly_chart(fig, use_container_width=True, key="results_confidence_distribution")

    st.markdown('<div class="section-title">Event Log</div>', unsafe_allow_html=True)
    st.dataframe(events_df, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-title">Exports</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download Processed Video", output_video.read_bytes(), file_name=output_video.name, mime="video/mp4")
    c2.download_button("Download Detection Events CSV", events_df.to_csv(index=False).encode("utf-8"), "detection_events.csv", "text/csv")
    c3.download_button("Download Summary CSV", summary_to_dataframe(summary).to_csv(index=False).encode("utf-8"), "summary.csv", "text/csv")


def render_future_features_note() -> None:
    """Show explicit boundary between current counting POC and future inspection capabilities."""

    st.markdown(
        """
        <div class="future-feature">
        This POC performs object counting from detections, tracking IDs, and line/zone crossings.
        Future model-specific features can add Defect Inspection, PPE Detection, Assembly Verification,
        and Missing Component Detection after suitable trained models are available.
        </div>
        """,
        unsafe_allow_html=True,
    )
