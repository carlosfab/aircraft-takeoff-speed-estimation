import numpy as np
import supervision as sv
from typing import Tuple


class Annotator:
    def __init__(self, color: sv.ColorPalette, resolution_wh: Tuple[int, int]) -> None:
        self.thickness = sv.calculate_dynamic_line_thickness(resolution_wh)
        self.box_annotator = sv.BoundingBoxAnnotator(
            color=color, thickness=self.thickness
        )
        self.label_annotator = sv.LabelAnnotator(color=color)
        # self.polygon_zone = sv.PolygonZone(SOURCE, frame_resolution_wh=resolution_wh)
        # self.trace_annotator = sv.TraceAnnotator(color=color, thickness=self.thickness)

    def annotate_frame(self, frame: np.array, detections: sv.Detections) -> np.array:
        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        return annotated_frame
