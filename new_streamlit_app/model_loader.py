import asyncio
import os
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks
import ultralytics.nn.modules.conv
import ultralytics.nn.modules.block  # ← tambahkan ini
import torch.nn.modules.container
import torch.nn.modules.conv
import torch.nn.modules.batchnorm
import torch.nn.modules.activation

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

    torch.serialization.add_safe_globals([
        ultralytics.nn.tasks.DetectionModel,
        ultralytics.nn.modules.conv.Conv,
        ultralytics.nn.modules.block.C2f,  # ← tambah ini!
        torch.nn.modules.container.Sequential,
        torch.nn.modules.activation.SiLU,
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.batchnorm.BatchNorm2d,
    ])

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelterbaik.pt")
    model = YOLO(model_path)
    return model
