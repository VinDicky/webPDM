import asyncio
import os
import sys
import torch
import inspect
from ultralytics import YOLO
import ultralytics.nn
import torch.nn

def load_yolov8_model():
    # Buat atau ambil event loop (untuk kompatibilitas Python 3.12)
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

    # Whitelist semua class dari ultralytics.nn dan torch.nn.modules.*
    safe_classes = []
    for module in [ultralytics.nn, torch.nn.modules]:
        for _, obj in inspect.getmembers(module, inspect.isclass):
            safe_classes.append(obj)

    torch.serialization.add_safe_globals(safe_classes)

    # Load model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelterbaik.pt")
    model = YOLO(model_path)
    return model
