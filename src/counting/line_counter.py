"""Virtual line/zone crossing logic for one-time object counting."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CountEvent:
    """An event emitted when a unique track crosses the configured line."""

    timestamp: str
    seconds: float
    frame_number: int
    track_id: int
    class_name: str
    confidence: float
    direction: str


@dataclass
class LineCounter:
    """Counts each tracked physical object once after crossing a virtual line."""

    mode: str
    line_position: float
    direction_filter: str
    frame_width: int
    frame_height: int
    counted_track_ids: set[int] = field(default_factory=set)
    forward_count: int = 0
    reverse_count: int = 0

    @property
    def total_count(self) -> int:
        """Total counted objects."""

        return self.forward_count + self.reverse_count

    def line_coordinate(self) -> int:
        """Return the active line x/y coordinate in pixels."""

        if self.mode == "Vertical Line":
            return int(self.frame_width * self.line_position)
        return int(self.frame_height * self.line_position)

    def crossing_direction(
        self,
        previous_center: tuple[int, int] | None,
        current_center: tuple[int, int],
    ) -> str | None:
        """Determine whether the current movement crosses the line."""

        if previous_center is None:
            return None
        if self.mode == "ROI Zone":
            return self._roi_crossing_direction(previous_center, current_center)
        line = self.line_coordinate()
        prev_x, prev_y = previous_center
        curr_x, curr_y = current_center

        if self.mode == "Vertical Line":
            if prev_x < line <= curr_x:
                return "Forward"
            if prev_x > line >= curr_x:
                return "Reverse"
        else:
            if prev_y < line <= curr_y:
                return "Forward"
            if prev_y > line >= curr_y:
                return "Reverse"
        return None

    def roi_bounds(self) -> tuple[int, int, int, int]:
        """Return a rectangular counting zone centered around line_position."""

        zone_height = max(24, int(self.frame_height * 0.12))
        center_y = int(self.frame_height * self.line_position)
        y1 = max(0, center_y - zone_height // 2)
        y2 = min(self.frame_height, center_y + zone_height // 2)
        return 0, y1, self.frame_width, y2

    def _roi_crossing_direction(
        self,
        previous_center: tuple[int, int],
        current_center: tuple[int, int],
    ) -> str | None:
        _, y1, _, y2 = self.roi_bounds()
        prev_inside = y1 <= previous_center[1] <= y2
        curr_inside = y1 <= current_center[1] <= y2
        if prev_inside or not curr_inside:
            return None
        return "Forward" if current_center[1] >= previous_center[1] else "Reverse"

    def should_count(self, track_id: int, direction: str | None) -> bool:
        """Return True when this track should be counted now."""

        if direction is None or track_id in self.counted_track_ids:
            return False
        if self.direction_filter != "Both" and self.direction_filter != direction:
            return False
        return True

    def register(self, track_id: int, direction: str) -> None:
        """Mark a track as counted and increment the matching counter."""

        self.counted_track_ids.add(track_id)
        if direction == "Forward":
            self.forward_count += 1
        else:
            self.reverse_count += 1
