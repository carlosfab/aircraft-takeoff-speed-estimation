"""Module for transforming points using a perspective transformation."""

import cv2
import numpy as np


class ViewTransformer:
    """
    A class for transforming points using a perspective transformation.

    This class takes source and destination points to calculate the perspective
    transformation matrix and provides a method to transform points using the
    calculated matrix.
    """

    def __init__(self, source_points: np.ndarray, destination_points: np.ndarray):
        """
        Initialize the ViewTransformer.

        Args:
            source_points (np.ndarray): Array of source points.
            destination_points (np.ndarray): Array of destination points.
        """
        self.source_points = source_points
        self.destination_points = destination_points
        self.transformation_matrix = cv2.getPerspectiveTransform(
            self.source_points, self.destination_points
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform the given points using the perspective transformation matrix.

        Args:
            points (np.ndarray): Array of points to be transformed.

        Returns:
            np.ndarray: Array of transformed points.
        """
        transformed_points = cv2.perspectiveTransform(
            points.reshape(-1, 1, 2), self.transformation_matrix
        )
        return transformed_points.reshape(-1, 2)
