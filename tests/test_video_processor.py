# test_video_processor.py

import os

import cv2
import numpy as np
import tempfile
import yaml
import pytest
import supervision as sv
from ultralytics import YOLO

from src.path import DATA_PATH, MODELS_PATH
from src.video_processor import VideoProcessor


@pytest.fixture
def video_path():
    return str(DATA_PATH / "santos_dumont_airport.mp4")


@pytest.fixture
def sample_frame():
    frame = cv2.imread(str(DATA_PATH / "first_frame_santos_dummont_airport.png"))
    return frame


@pytest.fixture
def model_weights():
    return str(MODELS_PATH / "yolov8n.pt")


@pytest.fixture
def polygon_points():
    return np.array([[642, 784], [1960, 881], [1146, 1009], [-161, 852]])


@pytest.fixture
def target_size():
    return {"width": 300, "height": 260}


@pytest.fixture
def video_processor(
    video_path,
    model_weights,
    polygon_points,
    target_size,
    confidence_threshold=0.25,
    nms_iou_threshold=0.15,
):
    return VideoProcessor(
        video_path,
        model_weights,
        polygon_points,
        target_size,
        confidence_threshold,
        nms_iou_threshold,
    )


def test_load_yolo_model(video_processor, video_path):
    assert video_processor.confidence_threshold == 0.25
    assert video_processor.nms_iou_threshold == 0.15
    assert video_processor.video_path == video_path
    assert isinstance(video_processor.model, YOLO)


def test_get_video_generator(video_processor):
    frame_generator = video_processor.get_video_generator()
    assert frame_generator is not None

    # Check if the generator yields frames
    frame_count = 0
    for frame in frame_generator:
        assert isinstance(frame, np.ndarray)
        assert frame.shape[0] > 0 and frame.shape[1] > 0
        frame_count += 1

    # Check if the generator yields the expected number of frames
    video_info = sv.VideoInfo.from_video_path(video_processor.video_path)
    assert frame_count == video_info.total_frames


def test_process_frame(video_processor, sample_frame):
    processed_frame = video_processor.process_frame(sample_frame)
    assert isinstance(processed_frame, np.ndarray)
    assert processed_frame.shape == sample_frame.shape
    assert processed_frame.dtype == sample_frame.dtype
    assert processed_frame is not sample_frame


def test_detect_object(video_processor, sample_frame):
    detections = video_processor.detect_objects(sample_frame)
    assert isinstance(detections, sv.Detections)
    assert len(detections) > 0


def test_video_processor_polygon_zone(video_processor):
    assert video_processor.polygon_zone is not None
    assert isinstance(video_processor.polygon_zone, sv.PolygonZone)


def test_save_processed_video(video_processor):
    video_processor.process_video(visualize=False)

    # Check if the output file exists
    assert os.path.exists(video_processor.output_path), "Output file does not exist"

    # Check if the output video has the same resolution, frames, and fps as the input video
    input_video_info = sv.VideoInfo.from_video_path(video_processor.video_path)
    output_video_info = sv.VideoInfo.from_video_path(video_processor.output_path)

    assert input_video_info.resolution_wh == output_video_info.resolution_wh
    assert input_video_info.fps == output_video_info.fps
    assert input_video_info.total_frames == output_video_info.total_frames
