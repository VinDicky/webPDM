import asyncio
import os
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks
import ultralytics.nn.modules.conv  # ← Tambahkan ini
import torch.nn.modules.container   # ← Untuk Sequential

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

    # Allowlist semua class yang diperlukan oleh PyTorch 2.6+
    torch.serialization.add_safe_globals([
        ultralytics.nn.tasks.DetectionModel,
        ultralytics.nn.modules.conv.Conv,
        torch.nn.modules.container.Sequential,
    ])

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelterbaik.pt")
    model = YOLO(model_path)
    return model
