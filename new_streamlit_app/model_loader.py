import asyncio
import os
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks  # pastikan import ini supaya class-nya dikenali

def load_yolov8_model():
    import sys
    try:
        # Python 3.12+ event loop handling
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
    
    # Tambahkan ini agar class DetectionModel di-allowlist saat loading model
    torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelterbaik.pt")

    # Load model dengan weights_only=False, ini mencegah error loading checkpoint
    model = YOLO(model_path, weights_only=False)

    return model
