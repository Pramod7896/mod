"""Track history management for visual trails and counting."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class TrackHistory:
    """Maintains recent center points for active tracks."""

    max_points: int = 48
    points: dict[int, deque[tuple[int, int]]] = field(default_factory=lambda: defaultdict(deque))

    def update(self, track_id: int, center: tuple[int, int]) -> tuple[int, int] | None:
        """Append the current center and return the previous center if available."""

        history = self.points[track_id]
        previous = history[-1] if history else None
        history.append(center)
        while len(history) > self.max_points:
            history.popleft()
        return previous

    def get(self, track_id: int) -> list[tuple[int, int]]:
        """Return trajectory points for a track."""

        return list(self.points.get(track_id, []))
