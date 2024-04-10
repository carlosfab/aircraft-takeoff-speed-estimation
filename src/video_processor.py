"""
Module for processing videos using object detection and tracking.

This module provides a VideoProcessor class that can process a video file,
detect objects, track them, and generate an annotated output video.
"""

from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import supervision as sv
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
from src.view_transformer import ViewTransformer
from src.speed_estimator import SpeedEstimator

from src.path import OUTPUT_PATH

PRIMARY_COLOR = sv.Color.from_hex("#FF103F")
SECONDARY_COLOR = sv.Color.from_hex("#016FB9")


class VideoProcessor:
    """
    A class for processing videos using object detection and tracking.

    This class takes a video file as input, applies object detection using a YOLO model,
    tracks the detected objects, and generates an annotated output video with bounding boxes,
    labels, and traces.
    """

    def __init__(
        self,
        video_path: str,
        model_weights: str,
        polygon_points: np.ndarray,
        target_size: Dict[str, int],
        confidence_threshold: float = 0.2,
        nms_iou_threshold: float = 0.2,
    ) -> None:
        """
        Initialize the VideoProcessor.

        Args:
            video_path (str): Path to the input video file.
            model_weights (str): Path to the YOLO model weights.
            polygon_points (np.ndarray): Array of polygon points for the zone.
            confidence_threshold (float, optional): Confidence threshold for object detection. Defaults to 0.2.
            nms_iou_threshold (float, optional): NMS IOU threshold for object detection. Defaults to 0.2.
        """
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
        self.target_points = self._get_target_coordinates(target_size)
        self.view_transformer = ViewTransformer(polygon_points, self.target_points)
        self.speed_estimator = SpeedEstimator(self.video_info.fps)

    def process_video(self, visualize: bool = True) -> None:
        """
        Process the video and generate the output video with annotations.

        Args:
            visualize (bool, optional): Whether to visualize the processed frames. Defaults to True.
        """
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

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame and return the annotated frame.

        Args:
            frame (np.ndarray): Input frame.

        Returns:
            np.ndarray: Annotated frame.
        """
        annotated_frame = frame.copy()

        detections = self.detect_objects(frame)

        points = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)

        if len(points) > 0:
            points = self.view_transformer.transform_points(points=points).astype(int)

            for tracker_id, [x, y] in zip(detections.tracker_id, points):
                self.speed_estimator.update_coordinates(tracker_id, np.array([x, y]))

        labels = []

        for tracker_in in detections.tracker_id:
            if (
                len(self.speed_estimator.coordinates[tracker_in])
                < self.speed_estimator.min_points
            ):
                labels.append(f"#{tracker_in}")
            else:
                speed = self.speed_estimator.estimate_speed(tracker_in, "knots")
                if speed is not None and speed > 5:
                    labels.append(f"#{tracker_in} {int(speed)} knots")
                else:
                    labels.append(f"#{tracker_in}")

        annotated_frame = self.round_box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels
        )
        annotated_frame = self.polygon_zone_annotator.annotate(annotated_frame)
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)

        return annotated_frame

    def get_video_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Get a generator that yields video frames.

        Returns:
            Generator[np.ndarray, None, None]: Generator yielding video frames.
        """
        frame_generator = sv.get_video_frames_generator(self.video_path)
        return frame_generator

    def detect_objects(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect objects in the given frame.

        Args:
            frame (np.ndarray): Input frame.

        Returns:
            sv.Detections: Detected objects.
        """
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 4]
        detections = detections[detections.confidence > self.confidence_threshold]
        detections = detections[self.polygon_zone.trigger(detections)]
        detections = detections.with_nms(self.nms_iou_threshold)
        detections = self.tracker.update_with_detections(detections)

        return detections

    def initialize_annotators(self) -> None:
        """Initialize the annotators."""
        self.round_box_annotator = sv.RoundBoxAnnotator(PRIMARY_COLOR)
        self.label_annotator = sv.LabelAnnotator(PRIMARY_COLOR)
        self.trace_annotator = sv.TraceAnnotator(SECONDARY_COLOR)
        self.polygon_zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.polygon_zone, color=SECONDARY_COLOR
        )

    def _get_target_coordinates(self, target_size: Dict[str, int]) -> np.ndarray:
        """Get the target points for the perspective transformation."""
        target_coordinates = np.array(
            [
                [0, 0],
                [target_size["width"], 0],
                [target_size["width"], target_size["height"]],
                [0, target_size["height"]],
            ]
        ).astype(np.float32)
        return target_coordinates
