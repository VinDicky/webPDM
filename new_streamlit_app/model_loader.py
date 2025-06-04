import os
import sys
import asyncio
import torch
from ultralytics import YOLO

def load_yolov8_model():
    try:
        if sys.version_info >= (3, 12):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        else:
            asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    from torch.serialization import add_safe_globals
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.conv import Conv, Concat
    from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF, DFL
    from ultralytics.nn.modules.head import Detect
    from torch.nn import BatchNorm2d, Sequential, SiLU
    from torch.nn.modules.conv import Conv2d
    from torch.nn.modules.container import ModuleList

    add_safe_globals([
        DetectionModel, Conv, Concat, C2f, Bottleneck,
        SPPF, DFL, Detect, BatchNorm2d, Conv2d,
        ModuleList, Sequential, SiLU  # ⬅ Penting!
    ])

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelterbaik.pt")
    model = YOLO(model_path)
    return model
