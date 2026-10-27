"""Shared RGB -> fixed square NCHW float32 preprocessing for YOLOv5 and PTQ."""

import cv2
import numpy as np
from PIL import Image

PREPROCESS_VERSION = "rgb-letterbox114-linear-nchw-div255-v1"


def prepare_image(image: Image.Image, size: int) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    ratio = min(size / height, size / width)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    if (width, height) != (resized_width, resized_height):
        rgb = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - resized_width) / 2, (size - resized_height) / 2
    padded = cv2.copyMakeBorder(rgb, round(dh - .1), round(dh + .1),
                                round(dw - .1), round(dw + .1), cv2.BORDER_CONSTANT,
                                value=(114, 114, 114))
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
