# test_speed_estimator.py

import numpy as np
import pytest

from src.speed_estimator import SpeedEstimator


@pytest.fixture
def speed_estimator():
    fps = 30
    return SpeedEstimator(fps)


def test_speed_estimator_initialization(speed_estimator):
    assert isinstance(speed_estimator, SpeedEstimator)
    assert speed_estimator.fps == 30


def test_update_coordinates(speed_estimator):
    tracker_id = 1
    points = np.array([100, 100])
    speed_estimator.update_coordinates(tracker_id, points)
    assert tracker_id in speed_estimator.coordinates
    assert len(speed_estimator.coordinates[tracker_id]) == 1


def test_update_multiple_tracker_coordinates(speed_estimator):
    for i in range(3):
        tracker_id = i
        points = np.random.randint(0, 100, size=(2,))
        speed_estimator.update_coordinates(tracker_id, points)

    assert len(speed_estimator.coordinates) == 3
    assert all(len(points) == 1 for points in speed_estimator.coordinates.values())


def test_estimate_speed_insufficient_points(speed_estimator):
    tracker_id = 1
    speed = speed_estimator.estimate_speed(tracker_id)
    assert speed is None


def test_calculate_speed_single_tracker(speed_estimator):
    tracker_id = 1
    num_frames = 30

    start_x = 0
    end_x = 300

    x_points = np.linspace(start=start_x, stop=end_x, num=num_frames).astype(int)
    y_points = np.random.randint(low=0, high=300, size=num_frames).astype(int)

    points = np.column_stack((x_points, y_points))

    for point in points:
        speed_estimator.update_coordinates(tracker_id, point)

    speed = speed_estimator.estimate_speed(tracker_id)

    # Expected speed calculation
    distance = end_x - start_x  # Distance covered      
    time = num_frames / speed_estimator.fps  # Time taken
    expected_speed = distance / time

    # Assert with a tolerance
    assert speed == pytest.approx(expected_speed, rel=1e-3)


def test_calculate_speed_multiple_frames_limited_buffer(speed_estimator):
    tracker_id = 1
    num_frames = 300
    start_x = 0
    end_x = 1000

    x_points = np.linspace(start=start_x, stop=end_x, num=num_frames).astype(int)
    y_points = np.random.randint(low=0, high=300, size=num_frames).astype(int)
    points = np.column_stack((x_points, y_points))

    # Check speed at different frame intervals
    frame_intervals = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]

    for i, point in enumerate(points):
        speed_estimator.update_coordinates(tracker_id, point)

        if i + 1 in frame_intervals:
            interval = i + 1
            speed = speed_estimator.estimate_speed(tracker_id)

            # Expected speed calculation
            buffer_size = speed_estimator.fps * 2
            start_index = max(0, interval - buffer_size)
            distance = x_points[i] - x_points[start_index]  # Distance covered
            time = (i - start_index + 1) / speed_estimator.fps  # Time taken
            expected_speed = distance / time

            # Assert with a tolerance
            assert speed == pytest.approx(expected_speed, rel=1e-3)
