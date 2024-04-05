import numpy as np

from src.path import CONFIG_PATH, DATA_PATH, MODELS_PATH, OUTPUT_PATH
from src.utils import load_config, parse_arguments
from src.video_processor import VideoProcessor

# Setting the video path
video_path = str(DATA_PATH / "santos_dumont_airport.mp4")
model_weights = str(MODELS_PATH / "yolov8x.pt")


def main():
    # Parse command-line arguments
    args = parse_arguments()
    config = load_config(CONFIG_PATH)

    source_points = np.array(config["source"]) if "source" in config else None
    target = config.get("target", {})
    confidence_threshold = config.get("detection", {}).get("confidence_threshold", 0.5)
    nms_iou_threshold = config.get("detection", {}).get("nms_iou_threshold", 0.5)

    video_processor = VideoProcessor(
        source_video_path=args.video,
        target_video_path=args.output,
        source_points=source_points,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
        model_weights="yolov8x.pt",
    )
    video_processor.process_video()


if __name__ == "__main__":
    main()
