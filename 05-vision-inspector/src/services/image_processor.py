"""Image preprocessing, enhancement, and ROI extraction.

The ``ImageProcessor`` class encapsulates deterministic image operations
that prepare raw camera frames for downstream defect detection.
"""

import cv2
import numpy as np


class ImageProcessor:
    """Stateless image preprocessor used throughout the inspection pipeline."""

    def preprocess(self, image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Resize, normalize to [0, 1], and convert to RGB.

        Parameters
        ----------
        image:
            Input BGR image from OpenCV.
        target_size:
            ``(width, height)`` for the output.

        Returns
        -------
        np.ndarray
            Float32 RGB image with values in [0, 1].
        """
        if image.size == 0:
            raise ValueError("Cannot preprocess an empty image")

        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    def extract_roi(self, image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Extract the region of interest.

        Parameters
        ----------
        image:
            Input image.
        mask:
            Binary mask (same spatial dimensions). Pixels > 0 are kept.
            ``None`` returns the full image unchanged.

        Returns
        -------
        np.ndarray
            Cropped image containing only the masked region.
        """
        if image.size == 0:
            raise ValueError("Cannot extract ROI from an empty image")

        if mask is None:
            return image.copy()

        if mask.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape[:2]} does not match image shape {image.shape[:2]}"
            )

        coords = cv2.findNonZero(mask)
        if coords is None:
            return np.zeros((0, 0, image.shape[2] if len(image.shape) == 3 else 1), dtype=image.dtype)
        x, y, w, h = cv2.boundingRect(coords)
        return image[y : y + h, x : x + w].copy()

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE histogram equalization and mild denoising.

        Parameters
        ----------
        image:
            Input BGR image.

        Returns
        -------
        np.ndarray
            Enhanced BGR image.
        """
        if image.size == 0:
            raise ValueError("Cannot enhance an empty image")

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        merged = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)

    def augment(self, image: np.ndarray, num_augmentations: int = 3) -> list[np.ndarray]:
        """Generate augmented variations of the input image.

        Produces brightness, rotation, and flip variations.

        Parameters
        ----------
        image:
            Input BGR image.
        num_augmentations:
            How many augmented copies to return (max 6).

        Returns
        -------
        list[np.ndarray]
            List of augmented images.
        """
        if image.size == 0:
            raise ValueError("Cannot augment an empty image")

        num_augmentations = max(1, min(num_augmentations, 6))
        results: list[np.ndarray] = []
        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        augmentations = [
            lambda img: cv2.convertScaleAbs(img, alpha=1.3, beta=20),
            lambda img: cv2.convertScaleAbs(img, alpha=0.7, beta=-20),
            lambda img: cv2.warpAffine(
                img,
                cv2.getRotationMatrix2D(center, 15, 1.0),
                (w, h),
                borderMode=cv2.BORDER_REFLECT,
            ),
            lambda img: cv2.flip(img, 1),
            lambda img: cv2.flip(img, 0),
            lambda img: cv2.GaussianBlur(img, (5, 5), 1.0),
        ]

        for i in range(num_augmentations):
            results.append(augmentations[i](image))
        return results

    def validate_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw image bytes and perform basic validation.

        Parameters
        ----------
        image_bytes:
            Raw JPEG/PNG bytes.

        Returns
        -------
        np.ndarray
            Decoded BGR image.

        Raises
        ------
        ValueError
            If the bytes do not represent a valid image.
        """
        if not image_bytes:
            raise ValueError("Empty image bytes")

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from provided bytes")

        h, w = image.shape[:2]
        if h < 1 or w < 1:
            raise ValueError(f"Invalid image dimensions: {w}x{h}")

        return image
