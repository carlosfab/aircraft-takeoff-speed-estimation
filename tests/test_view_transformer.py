# test_view_transformer.py

import cv2
import numpy as np
import pytest

from src.view_transformer import ViewTransformer


@pytest.fixture
def source_points():
    return np.array(
        [[642, 784], [1960, 881], [1146, 1009], [-161, 852]], dtype=np.float32
    )


@pytest.fixture
def destination_points():
    return np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)


@pytest.fixture
def view_transformer(source_points, destination_points):
    return ViewTransformer(source_points, destination_points)


def test_view_transformer_initialization(
    view_transformer, source_points, destination_points
):
    assert isinstance(view_transformer, ViewTransformer)
    assert view_transformer.source_points.shape == source_points.shape
    assert view_transformer.destination_points.shape == destination_points.shape


def test_view_transformer_transform_points(view_transformer):
    points = np.array([[100, 100], [200, 200], [300, 300]], dtype=np.float32)
    transformed_points = view_transformer.transform_points(points)
    assert isinstance(transformed_points, np.ndarray)
    assert transformed_points.shape == points.shape


def test_view_transformer_extreme_points(
    view_transformer, source_points, destination_points
):
    extreme_points = source_points
    transformed_extreme_points = view_transformer.transform_points(extreme_points)
    assert np.allclose(transformed_extreme_points, destination_points, atol=1e-3)


def test_view_transformer_outside_points(view_transformer):
    outside_points = np.array([[-1000, -1000], [3000, 3000]], dtype=np.float32)
    transformed_outside_points = view_transformer.transform_points(outside_points)
    assert not np.logical_and(
        np.all(transformed_outside_points >= 0),
        np.all(transformed_outside_points <= 100),
    ).any()


def test_view_transformer_reverse_transformation(view_transformer):
    points = np.array([[100, 100], [200, 200], [300, 300]], dtype=np.float32)
    transformed_points = view_transformer.transform_points(points)
    reverse_transformation_matrix = cv2.getPerspectiveTransform(
        view_transformer.destination_points, view_transformer.source_points
    )
    reverse_transformed_points = cv2.perspectiveTransform(
        transformed_points.reshape(-1, 1, 2), reverse_transformation_matrix
    ).reshape(-1, 2)
    assert np.allclose(points, reverse_transformed_points, atol=1e-3)
