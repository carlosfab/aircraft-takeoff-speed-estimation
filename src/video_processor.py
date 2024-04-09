# video_processor.py

from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm.auto import tqdm
from ultralytics import YOLO

from src.path import OUTPUT_PATH

PRIMARY_COLOR = sv.Color.from_hex("#FF103F")
SECONDARY_COLOR = sv.Color.from_hex("#016FB9")


class VideoProcessor:
    def __init__(
        self,
        video_path: str,
        model_weights: str,
        polygon_points: np.ndarray,
        confidence_threshold: float = 0.2,
        nms_iou_threshold: float = 0.2,
    ) -> None:
        self.video_path = video_path
        self.model = YOLO(model_weights)
        self.video_info = sv.VideoInfo.from_video_path(video_path)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.polygon_zone = sv.PolygonZone(
            polygon_points, frame_resolution_wh=self.video_info.resolution_wh
        )
        self.tracker = sv.ByteTrack()
        self.initialize_annotators()

    def process_video(self, visualize: bool = True):
        frame_generator = self.get_video_generator()

        # Create the output file path with "output_" prefix and located in DATA_PATH
        input_file_name = Path(self.video_path).stem
        output_file_name = f"output_{input_file_name}.mp4"
        self.output_path = str(OUTPUT_PATH / output_file_name)

        with sv.VideoSink(self.output_path, self.video_info) as sink:
            for frame in tqdm(
                frame_generator,
                total=self.video_info.total_frames,
                desc="Processing frames",
            ):
                annotated_frame = self.process_frame(frame)
                sink.write_frame(annotated_frame)

                if visualize:
                    cv2.imshow("Video", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    def process_frame(self, frame):
        annotated_frame = frame.copy()

        detections = self.detect_objects(frame)

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

        annotated_frame = self.round_box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels
        )
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)

        return annotated_frame

    def get_video_generator(self):
        frame_generator = sv.get_video_frames_generator(self.video_path)
        return frame_generator

    def detect_objects(self, frame):
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 4]
        detections = detections[detections.confidence > self.confidence_threshold]
        detections = detections.with_nms(self.nms_iou_threshold)
        detections = self.tracker.update_with_detections(detections)

        return detections

    def initialize_annotators(self):
        self.round_box_annotator = sv.RoundBoxAnnotator(PRIMARY_COLOR)
        self.label_annotator = sv.LabelAnnotator(PRIMARY_COLOR)
        self.trace_annotator = sv.TraceAnnotator(SECONDARY_COLOR)
        self.polygon_zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.polygon_zone, color=SECONDARY_COLOR
        )
