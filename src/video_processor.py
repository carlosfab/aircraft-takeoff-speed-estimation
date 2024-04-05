import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

PRIMARY_COLOR = sv.Color.from_hex("#FF103F")
SECONDARY_COLOR = sv.Color.from_hex("#016FB9")


class VideoProcessor:
    def __init__(
        self,
        source_video_path: str,
        model_weights: str,
        source_points: np.ndarray,
        confidence_threshold: float,
        nms_iou_threshold: float,
        target_video_path: str = None,
    ):
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.source_points = source_points
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.model = YOLO(model_weights)
        self.tracker = sv.ByteTrack(
            lost_track_buffer=60, frame_rate=self.video_info.fps
        )

        # Initialize model and processing tools
        self.annotators = []  # Initialize the list of annotators
        self.initialize_annotators()

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(self.source_video_path)

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in frame_generator:
                    processed_frame = self.process_frame(frame)
                    sink.write_frame(processed_frame)
                    self.plot_frame(processed_frame)

    def process_frame(self, frame: np.ndarray, imgsz: int = None) -> np.ndarray:
        results = self.model(frame)[0]

        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.confidence > self.confidence_threshold]
        detections = detections[detections.class_id == 4]
        detections = detections.with_nms(self.nms_iou_threshold)
        detections = detections[self.polygon_zone.trigger(detections)]
        detections = self.tracker.update_with_detections(detections)

        transformed_frame = self.annotate_frame(frame, detections)

        return transformed_frame

    def plot_frame(self, frame: np.ndarray):
        cv2.imshow("Processed Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise StopIteration

    def initialize_annotators(self):
        """Initializes annotators for visualizing detections and tracking on video frames."""
        self.polygon_zone = sv.PolygonZone(
            self.source_points, self.video_info.resolution_wh
        )

        polygon_zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.polygon_zone,
            color=SECONDARY_COLOR,
            display_in_zone_count=False,
        )

        round_box_annotator = sv.RoundBoxAnnotator(color=PRIMARY_COLOR)
        label_annotator = sv.LabelAnnotator(color=PRIMARY_COLOR)

        # Add the annotators to the list
        self.annotators.extend(
            [polygon_zone_annotator, round_box_annotator, label_annotator]
        )

    def annotate_frame(self, frame: np.ndarray, detections) -> np.ndarray:
        transformed_frame = frame.copy()

        labels = [f"#{tracker_id}" for _, _, _, _, tracker_id, _ in detections]

        for annotator in self.annotators:
            if isinstance(annotator, sv.LabelAnnotator):
                print(detections)
                transformed_frame = annotator.annotate(
                    transformed_frame, detections, labels
                )

            elif isinstance(annotator, sv.PolygonZoneAnnotator):
                transformed_frame = annotator.annotate(transformed_frame)

            elif isinstance(annotator, sv.RoundBoxAnnotator):
                transformed_frame = annotator.annotate(transformed_frame, detections)

        return transformed_frame
