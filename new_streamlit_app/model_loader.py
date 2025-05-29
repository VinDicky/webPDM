import asyncio
from ultralytics import YOLO
import os

def load_yolov8_model():
    import sys
    try:
        # For Python 3.12 compatibility, use this approach to get or create event loop
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
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelterbaik.pt")
    model = YOLO(model_path)
    return model
