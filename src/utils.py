import argparse
from src.path import CONFIG_PATH, DATA_PATH, MODELS_PATH, OUTPUT_PATH
import supervision as sv

COLORS = sv.ColorPalette.DEFAULT


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    ap = argparse.ArgumentParser(description="Speed estimation using YOLOv8")
    ap.add_argument(
        "-v",
        "--video",
        required=False,
        help="Path to input video",
        type=str,
        default=f"{DATA_PATH}/SDU_01.mp4",
    )
    ap.add_argument(
        "-o",
        "--output",
        required=False,
        help="Path to output video",
        type=str,
        default=f"{OUTPUT_PATH}/output.mp4",
    )
    ap.add_argument(
        "-m",
        "--model",
        required=False,
        help="Path to YOLO model weights",
        type=str,
        default=f"{MODELS_PATH}/yolov8x.pt",
    )
    return ap.parse_args()
