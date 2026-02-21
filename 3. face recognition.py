"""
Professional Face Detection with auto-resize for large images.
Compatible with pylint & flake8.
"""

import logging
import os
from typing import List, Tuple

import cv2
import numpy as np


MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIG_FILE = "deploy.prototxt"
CONFIDENCE_THRESHOLD = 0.35
MAX_WIDTH = 1200  # Resize images wider than this


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_model() -> cv2.dnn_Net:
    """Load the DNN model."""
    if not os.path.isfile(CONFIG_FILE):
        raise FileNotFoundError("deploy.prototxt not found")

    if not os.path.isfile(MODEL_FILE):
        raise FileNotFoundError("caffemodel not found")

    return cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)


def resize_if_large(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Resize image if too large while keeping aspect ratio."""
    height, width = image.shape[:2]

    if width > MAX_WIDTH:
        scale = MAX_WIDTH / float(width)
        resized = cv2.resize(image, None, fx=scale, fy=scale)
        return resized, scale

    return image, 1.0


def detect_faces(
    net: cv2.dnn_Net,
    image: np.ndarray
) -> List[Tuple[int, int, int, int]]:
    """Detect faces in image."""
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    faces = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array(
                [width, height, width, height]
            )
            faces.append(tuple(box.astype(int)))

    return faces


def process_images(image_paths: List[str]) -> None:
    """Process images safely."""
    net = load_model()

    for image_path in image_paths:
        if not os.path.isfile(image_path):
            logging.warning("Missing image: %s", image_path)
            continue

        image = cv2.imread(image_path)

        if image is None:
            logging.error("Failed to load: %s", image_path)
            continue

        resized_image, scale = resize_if_large(image)

        faces = detect_faces(net, resized_image)

        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(
                resized_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        logging.info(
            "%s -> Faces detected: %d",
            image_path,
            len(faces)
        )

        cv2.imshow("Face Detection", resized_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def main() -> None:
    """Main function."""
    image_files = [
        "pic 1.jpg",
        "pic 2.jpg",
        "image.png",
        "image copy.png",
        "image copy 2.png",
        "1.jpg",
        "1.png"
    ]

    process_images(image_files)


if __name__ == "__main__":
    main()