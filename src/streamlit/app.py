import cv2
import numpy as np

import streamlit as st

# Set the initial SOURCE and TARGET values
SOURCE = np.array([[642, 784], [1960, 881], [1146, 1009], [-161, 852]])

TARGET_WIDTH = 300
TARGET_HEIGHT = 260


def transform_perspective(image, source, target_width, target_height):
    target = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(source.astype(np.float32), target)
    transformed_image = cv2.warpPerspective(image, M, (target_width, target_height))
    return transformed_image


def draw_lines_on_image(image, source_points):
    lines_image = image.copy()
    for i in range(4):
        cv2.line(
            lines_image,
            tuple(source_points[i]),
            tuple(source_points[(i + 1) % 4]),
            (0, 255, 0),
            2,
        )
    return lines_image


def main():
    st.set_page_config(layout="wide")
    st.title("Interactive Perspective Transformation")

    # Sidebar controls
    st.sidebar.title("Controls")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Sidebar sliders for SOURCE coordinates
        st.sidebar.subheader("Adjust SOURCE Coordinates")
        source_points = []
        for i in range(4):
            x = st.sidebar.slider(
                f"Point {i+1} X",
                -200,
                image.shape[1] + 100,
                SOURCE[i][0],
                key=f"point{i}_x",
            )
            y = st.sidebar.slider(
                f"Point {i+1} Y",
                -200,
                image.shape[0] + 100,
                SOURCE[i][1],
                key=f"point{i}_y",
            )
            source_points.append([x, y])

        # Sidebar sliders for TARGET dimensions
        st.sidebar.subheader("Adjust TARGET Dimensions")
        target_width = st.sidebar.slider("Target Width", 100, 500, TARGET_WIDTH)
        target_height = st.sidebar.slider("Target Height", 100, 500, TARGET_HEIGHT)

        # Draw lines on the original image
        image_with_lines = draw_lines_on_image(image, source_points)

        # Perform perspective transformation
        transformed_image = transform_perspective(
            image, np.array(source_points), target_width, target_height
        )

        # Display the original image with lines and the transformed image side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image with Lines")
            st.image(image_with_lines, channels="BGR")
        with col2:
            st.subheader("Transformed Image")
            st.image(transformed_image, channels="BGR")


if __name__ == "__main__":
    main()
