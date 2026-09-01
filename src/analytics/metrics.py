"""Manufacturing counting analytics calculated from real processing events."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SummaryMetrics:
    """Final production metrics."""

    total_count: int
    forward_count: int
    reverse_count: int
    average_confidence: float
    average_throughput: float
    peak_throughput: float
    lowest_throughput: float
    average_spacing_seconds: float | None
    processing_time_seconds: float
    elapsed_video_seconds: float


def events_to_dataframe(events: list[dict]) -> pd.DataFrame:
    """Convert event dictionaries into a stable DataFrame schema."""

    columns = ["timestamp", "seconds", "frame_number", "track_id", "class", "confidence", "direction"]
    if not events:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(events, columns=columns)


def build_timeline(events_df: pd.DataFrame, duration_seconds: float) -> pd.DataFrame:
    """Build per-second cumulative counts and object/minute rate."""

    total_seconds = max(1, int(duration_seconds) + 1)
    timeline = pd.DataFrame({"second": range(total_seconds)})
    if events_df.empty:
        timeline["count"] = 0
    else:
        counts = events_df.assign(second=events_df["seconds"].astype(int)).groupby("second").size()
        timeline["count"] = timeline["second"].map(counts).fillna(0).astype(int)
    timeline["cumulative_count"] = timeline["count"].cumsum()
    timeline["objects_per_minute"] = timeline["count"].rolling(window=60, min_periods=1).sum()
    return timeline


def calculate_summary(
    events_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    forward_count: int,
    reverse_count: int,
    processing_time_seconds: float,
    elapsed_video_seconds: float,
) -> SummaryMetrics:
    """Calculate final metrics from actual count events."""

    total_count = int(len(events_df))
    average_confidence = float(events_df["confidence"].mean()) if not events_df.empty else 0.0
    elapsed_minutes = max(elapsed_video_seconds / 60, 1 / 60)
    average_throughput = total_count / elapsed_minutes
    peak = float(timeline_df["objects_per_minute"].max()) if not timeline_df.empty else 0.0
    positive_rates = timeline_df.loc[timeline_df["objects_per_minute"] > 0, "objects_per_minute"]
    lowest = float(positive_rates.min()) if not positive_rates.empty else 0.0
    spacing = None
    if len(events_df) > 1:
        spacing = float(events_df["seconds"].sort_values().diff().dropna().mean())
    return SummaryMetrics(
        total_count=total_count,
        forward_count=forward_count,
        reverse_count=reverse_count,
        average_confidence=average_confidence,
        average_throughput=float(average_throughput),
        peak_throughput=peak,
        lowest_throughput=lowest,
        average_spacing_seconds=spacing,
        processing_time_seconds=float(processing_time_seconds),
        elapsed_video_seconds=float(elapsed_video_seconds),
    )


def summary_to_dataframe(summary: SummaryMetrics) -> pd.DataFrame:
    """Create an exportable one-row summary table."""

    return pd.DataFrame(
        [
            {
                "total_objects": summary.total_count,
                "forward_count": summary.forward_count,
                "reverse_count": summary.reverse_count,
                "average_confidence": summary.average_confidence,
                "average_throughput_objects_min": summary.average_throughput,
                "peak_throughput_objects_min": summary.peak_throughput,
                "lowest_throughput_objects_min": summary.lowest_throughput,
                "average_spacing_seconds": summary.average_spacing_seconds,
                "processing_time_seconds": summary.processing_time_seconds,
                "elapsed_video_seconds": summary.elapsed_video_seconds,
            }
        ]
    )
