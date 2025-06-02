import os
import sys
import asyncio
import torch
from ultralytics import YOLO

def load_yolov8_model():
    # Untuk Python 3.12+ event loop compatibility
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

    # Tambahkan class-class yang diizinkan secara eksplisit untuk menghindari error weights_only
    from torch.serialization import add_safe_globals
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.modules.block import C2f, Bottleneck
    from torch.nn import BatchNorm2d
    from torch.nn.modules.conv import Conv2d
    from torch.nn.modules.container import ModuleList

    add_safe_globals([Conv, C2f, Bottleneck, BatchNorm2d, Conv2d, ModuleList])

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelterbaik.pt")
    model = YOLO(model_path)
    return model
