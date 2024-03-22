import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from src.annotator import Annotator
from src.utils import COLORS

class VideoProcessor:
    def __init__(
        self,
        source_video_path: str,
        target_video_path: str,
        model_weights: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ) -> None:
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(model_weights)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(
            video_path=source_video_path
        )
        self.annotator = Annotator(
            color=COLORS, resolution_wh=self.video_info.resolution_wh
        )

    def process_video(self) -> None:
        frame_generator = sv.get_video_frames_generator(self.source_video_path)

        for frame in frame_generator:
            processed_frame = self.process_frame(frame)
            cv2.imshow('frame', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def process_frame(self, frame: np.array) -> np.array:
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        processed_frame = self.annotator.annotate_frame(frame, detections)

        return processed_frame