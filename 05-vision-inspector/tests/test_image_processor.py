"""Tests for the ImageProcessor service.

Every test creates synthetic numpy images — no external files required.
"""

import struct

import cv2
import numpy as np
import pytest

from src.services.image_processor import ImageProcessor


@pytest.fixture()
def processor() -> ImageProcessor:
    return ImageProcessor()


@pytest.fixture()
def sample_bgr() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def sample_gray() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640), dtype=np.uint8)


class TestPreprocess:
    def test_resize_dimensions(self, processor: ImageProcessor, sample_bgr: np.ndarray) -> None:
        result = processor.preprocess(sample_bgr, (320, 240))
        assert result.shape == (240, 320, 3)

    def test_normalisation_range(self, processor: ImageProcessor, sample_bgr: np.ndarray) -> None:
        result = processor.preprocess(sample_bgr, (320, 240))
        assert result.dtype == np.float32
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_rgb_conversion(self, processor: ImageProcessor, sample_bgr: np.ndarray) -> None:
        """Output should be RGB (channel 0 correlates with input red)."""
        sample_bgr[:, :, 2] = 250  # set BGR red channel high
        sample_bgr[:, :, 0] = 10   # set BGR blue channel low
        result = processor.preprocess(sample_bgr, (320, 240))
        # channel 0 of output is R (was BGR channel 2)
        assert float(result[:, :, 0].mean()) > 0.9

    def test_empty_image_raises(self, processor: ImageProcessor) -> None:
        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        with pytest.raises(ValueError, match="empty"):
            processor.preprocess(empty, (100, 100))


class TestExtractROI:
    def test_no_mask_returns_copy(self, processor: ImageProcessor, sample_bgr: np.ndarray) -> None:
        result = processor.extract_roi(sample_bgr, None)
        np.testing.assert_array_equal(result, sample_bgr)

    def test_mask_crops(self, processor: ImageProcessor, sample_bgr: np.ndarray) -> None:
        mask = np.zeros(sample_bgr.shape[:2], dtype=np.uint8)
        mask[100:200, 100:300] = 255
        result = processor.extract_roi(sample_bgr, mask)
        assert result.shape[0] <= 100 + 1
        assert result.shape[1] <= 200 + 1

    def test_mismatched_mask_raises(self, processor: ImageProcessor, sample_bgr: np.ndarray) -> None:
        wrong_mask = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="Mask shape"):
            processor.extract_roi(sample_bgr, wrong_mask)

    def test_all_zeros_mask(self, processor: ImageProcessor, sample_bgr: np.ndarray) -> None:
        mask = np.zeros(sample_bgr.shape[:2], dtype=np.uint8)
        result = processor.extract_roi(sample_bgr, mask)
        assert result.size == 0 or result.shape[0] == 0


class TestEnhance:
    def test_output_shape_unchanged(self, processor: ImageProcessor, sample_bgr: np.ndarray) -> None:
        result = processor.enhance(sample_bgr)
        assert result.shape == sample_bgr.shape

    def test_output_dtype(self, processor: ImageProcessor, sample_bgr: np.ndarray) -> None:
        result = processor.enhance(sample_bgr)
        assert result.dtype == np.uint8

    def test_empty_raises(self, processor: ImageProcessor) -> None:
        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        with pytest.raises(ValueError, match="empty"):
            processor.enhance(empty)


class TestAugment:
    def test_count(self, processor: ImageProcessor, sample_bgr: np.ndarray) -> None:
        result = processor.augment(sample_bgr, num_augmentations=4)
        assert len(result) == 4

    def test_shapes_match(self, processor: ImageProcessor, sample_bgr: np.ndarray) -> None:
        results = processor.augment(sample_bgr, 2)
        for img in results:
            assert img.shape == sample_bgr.shape

    def test_clamps_range(self, processor: ImageProcessor, sample_bgr: np.ndarray) -> None:
        results = processor.augment(sample_bgr, 6)
        for img in results:
            assert img.min() >= 0
            assert img.max() <= 255

    def test_empty_raises(self, processor: ImageProcessor) -> None:
        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        with pytest.raises(ValueError, match="empty"):
            processor.augment(empty)


class TestValidateImage:
    def test_valid_jpeg(self, processor: ImageProcessor) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        result = processor.validate_image(buf.tobytes())
        assert result.shape == (100, 100, 3)

    def test_valid_png(self, processor: ImageProcessor) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        result = processor.validate_image(buf.tobytes())
        assert result.shape == (100, 100, 3)

    def test_empty_bytes_raises(self, processor: ImageProcessor) -> None:
        with pytest.raises(ValueError, match="Empty image bytes"):
            processor.validate_image(b"")

    def test_garbage_raises(self, processor: ImageProcessor) -> None:
        with pytest.raises(ValueError, match="Could not decode"):
            processor.validate_image(b"\x00\x01\x02\x03\x04")
