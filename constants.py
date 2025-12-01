import os

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# CUDA defaults (can be overridden via env LLAVA_CUDA_DEVICES)
# Comma-separated device list, e.g., "0" to pin to a specific GPU.
DEFAULT_CUDA_DEVICES = os.getenv("LLAVA_CUDA_DEVICES", "0")

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
