import os
import sys
import asyncio
from ultralytics import YOLO

def load_yolov8_model():
    try:
        # Setup asyncio event loop untuk Python >= 3.12
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

    # Path ke file model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelterbaik.pt")

    # Load YOLOv8 model
    model = YOLO(model_path)

    return model
