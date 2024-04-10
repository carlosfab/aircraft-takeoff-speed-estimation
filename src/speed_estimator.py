"""Module for estimating speed based on object coordinates."""

from collections import defaultdict, deque

import numpy as np


class SpeedEstimator:
    """Class for estimating speed based on object coordinates."""

    def __init__(self, fps: int):
        """
        Initialize the SpeedEstimator.

        Args:
            fps (int): Frames per second of the video.
        """
        self.fps = fps
        self.coordinates = defaultdict(lambda: deque(maxlen=self.fps * 2))
        self.min_points = self.fps // 2

    def update_coordinates(self, tracker_id: int, points: np.ndarray):
        """
        Update the coordinates for a given tracker ID.

        Args:
            tracker_id (int): ID of the tracked object.
            points (np.ndarray): Array of coordinates for the object.
        """
        self.coordinates[tracker_id].append(points)

    def estimate_speed(self, tracker_id: int, unit="m/s"):
        """
        Estimate the speed for a given tracker ID.

        Args:
            tracker_id (int): ID of the tracked object.
            unit (str): Unit of the speed (default: "m/s").
                Supported units: "m/s", "km/h", "knots".

        Returns:
            float: Estimated speed in the specified unit.
            None: If there are not enough points to estimate the speed.

        Raises:
            ValueError: If an invalid unit is provided.
        """
        points = self.coordinates[tracker_id]

        if len(points) < self.min_points:
            return None

        coordinate_start_x = points[0][0]
        coordinate_end_x = points[-1][0]
        distance = abs(coordinate_start_x - coordinate_end_x)
        time = len(points) / self.fps

        if unit == "m/s":
            speed = distance / time
        elif unit == "km/h":
            speed = (distance / time) * 3.6
        elif unit == "knots":
            speed = (distance / time) * 1.94384
        else:
            raise ValueError(f"Invalid unit: {unit}")

        return speed
