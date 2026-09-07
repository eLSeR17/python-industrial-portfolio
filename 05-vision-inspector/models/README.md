# Models

Place trained YOLOv8 model weights (`.pt`) in this directory.

Models are loaded at startup when `VISION_MODEL_PATH` is set.
If no model is provided, the service falls back to classical
computer-vision techniques (Canny edges + contour analysis).
