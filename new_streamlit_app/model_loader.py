import os
import sys
import torch
import asyncio
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.model import YOLO

def load_yolov8_model():
    # Buat event loop jika diperlukan (untuk Python 3.12+)
    try:
        if sys.version_info >= (3, 12):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        else:
            asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Path ke model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelterbaik.pt")

    # Bypass weights_only dengan cara aman
    with torch.serialization.safe_open(model_path, weights_only=False) as f:
        ckpt = f.load()

    # Buat model dari checkpoint yang dimuat
    model = YOLO()
    model.model.load_state_dict(ckpt['model'].state_dict())  # load model weights dari dict
    return model
