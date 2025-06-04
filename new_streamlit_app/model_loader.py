import os
import sys
import asyncio
import torch
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

    # Import semua kelas custom yang digunakan model
    from torch.serialization import add_safe_globals
    from ultralytics.nn.modules.conv import Conv, Concat
    from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF, DFL
    from ultralytics.nn.modules.head import Detect
    from torch.nn import BatchNorm2d
    from torch.nn.modules.conv import Conv2d
    from torch.nn.modules.container import ModuleList

    # Izinkan global class yang dibutuhkan model untuk deserialisasi
    add_safe_globals([Conv, Concat, C2f, Bottleneck, SPPF, DFL, Detect, BatchNorm2d, Conv2d, ModuleList])

    # Path ke file model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelterbaik.pt")

    # Load YOLOv8 model
    model = YOLO(model_path)

    return model
