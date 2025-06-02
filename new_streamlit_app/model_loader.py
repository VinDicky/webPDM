import asyncio
import os
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks
import torch.nn.modules.container  # <- untuk Sequential

def load_yolov8_model():
    import sys
    try:
        if sys.version_info >= (3, 12):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
        else:
            asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    # Allowlist semua class yang dibutuhkan untuk load model
    torch.serialization.add_safe_globals([
        ultralytics.nn.tasks.DetectionModel,
        torch.nn.modules.container.Sequential,
    ])

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelterbaik.pt")
    model = YOLO(model_path)
    return model
