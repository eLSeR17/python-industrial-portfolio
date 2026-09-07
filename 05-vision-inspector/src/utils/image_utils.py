"""Low-level image utilities — encoding, histograms, synthetic data.

All functions are pure (no side-effects beyond returning values) and
operate on raw NumPy arrays produced by OpenCV.
"""

import base64
import math
import uuid

import cv2
import numpy as np


def encode_image_base64(image: np.ndarray) -> str:
    """Encode a NumPy image to a Base64 JPEG string.

    Parameters
    ----------
    image:
        BGR or RGB image array.

    Returns
    -------
    str
        Base64-encoded JPEG bytes.
    """
    if image.size == 0:
        raise ValueError("Cannot encode an empty image")
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buffer).decode("utf-8")


def decode_image_base64(base64_str: str) -> np.ndarray:
    """Decode a Base64 string back into a BGR NumPy image.

    Parameters
    ----------
    base64_str:
        Base64-encoded image bytes (JPEG, PNG, etc.).

    Returns
    -------
    np.ndarray
        Decoded BGR image.

    Raises
    ------
    ValueError
        If the decoded bytes are not a valid image.
    """
    raw = base64.b64decode(base64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data — cv2.imdecode returned None")
    return image


def calculate_histogram(image: np.ndarray) -> dict[str, list[int]]:
    """Compute per-channel histograms for an image.

    Parameters
    ----------
    image:
        BGR or grayscale image.

    Returns
    -------
    dict
        Mapping of channel name to histogram (256 bins).
    """
    if image.size == 0:
        raise ValueError("Cannot compute histogram of an empty image")

    if len(image.shape) == 2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return {"gray": [int(v) for v in hist.flatten()]}

    channels = ["blue", "green", "red"]
    result: dict[str, list[int]] = {}
    for idx, name in enumerate(channels):
        hist = cv2.calcHist([image], [idx], None, [256], [0, 256])
        result[name] = [int(v) for v in hist.flatten()]
    return result


def resize_to_fit(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Resize an image to fit within the given bounds, preserving aspect ratio.

    Parameters
    ----------
    image:
        Input image.
    max_width:
        Maximum output width in pixels.
    max_height:
        Maximum output height in pixels.

    Returns
    -------
    np.ndarray
        Resized image (may be smaller than the bounds).
    """
    if image.size == 0:
        raise ValueError("Cannot resize an empty image")
    h, w = image.shape[:2]
    if w <= max_width and h <= max_height:
        return image.copy()
    scale = min(max_width / w, max_height / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def create_synthetic_defect_image(
    width: int = 640,
    height: int = 480,
    defect_type: str = "dent",
    position: tuple[int, int] | None = None,
) -> np.ndarray:
    """Generate a test image containing a synthetic defect.

    Useful for unit tests and demos without requiring real manufacturing
    imagery.

    Parameters
    ----------
    width:
        Image width in pixels.
    height:
        Image height in pixels.
    defect_type:
        One of ``"scratch"``, ``"dent"``, ``"crack"``, ``"stain"``.
    position:
        ``(cx, cy)`` center of the defect; ``None`` picks a random spot.

    Returns
    -------
    np.ndarray
        BGR image with the synthetic defect drawn on a uniform background.
    """
    img = np.full((height, width, 3), fill_value=180, dtype=np.uint8)

    rng = np.random.RandomState(hash(defect_type) % (2**31))
    if position is None:
        cx = int(rng.uniform(width * 0.2, width * 0.8))
        cy = int(rng.uniform(height * 0.2, height * 0.8))
    else:
        cx, cy = position

    defect_type_lower = defect_type.lower()

    if defect_type_lower == "scratch":
        length = int(min(width, height) * 0.25)
        angle = int(rng.uniform(0, 180))
        pt1 = (cx - length // 2, cy)
        pt2 = (cx + length // 2, cy)
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle), 1.0)
        p1 = (int(M[0, 0] * pt1[0] + M[0, 1] * pt1[1] + M[0, 2]),
               int(M[1, 0] * pt1[0] + M[1, 1] * pt1[1] + M[1, 2]))
        p2 = (int(M[0, 0] * pt2[0] + M[0, 1] * pt2[1] + M[0, 2]),
               int(M[1, 0] * pt2[0] + M[1, 1] * pt2[1] + M[1, 2]))
        cv2.line(img, p1, p2, color=(30, 30, 30), thickness=2)

    elif defect_type_lower == "dent":
        axes = (int(width * 0.06), int(height * 0.04))
        cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, color=(100, 100, 100), thickness=-1)

    elif defect_type_lower == "crack":
        pts: list[tuple[int, int]] = [(cx, cy)]
        x, y = cx, cy
        for _ in range(12):
            x += int(rng.randint(-15, 16))
            y += int(rng.randint(5, 18))
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], color=(20, 20, 20), thickness=2)

    elif defect_type_lower == "stain":
        radius = int(min(width, height) * 0.05)
        cv2.circle(img, (cx, cy), radius, color=(70, 60, 90), thickness=-1)

    else:
        raise ValueError(f"Unknown defect_type: {defect_type!r}")

    noise = rng.randint(0, 10, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img
