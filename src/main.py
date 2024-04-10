import yaml
import cv2
import numpy as np
from src.video_processor import VideoProcessor
from src.speed_estimator import SpeedEstimator
from src.view_transformer import ViewTransformer
from src.utils import load_config, parse_arguments
from src.path import DATA_PATH, MODELS_PATH, OUTPUT_PATH, CONFIG_PATH


def main():
    args = parse_arguments()
    config = load_config(CONFIG_PATH)

    video_path = args.video
    model_weights = args.model
    confidence_threshold = config["confidence_threshold"]
    nms_iou_threshold = config["nms_iou_threshold"]
    source = np.array(config["source"])
    target = config["target"]

    video_processor = VideoProcessor(
        video_path=video_path,
        model_weights=model_weights,
        polygon_points=source,
        target_size=target,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )
    video_processor.process_video()


if __name__ == "__main__":
    main()
