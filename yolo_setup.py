import torch

from ultralytics import YOLO

def load_yolov8_model():
    # Load the YOLOv8 model
    model = YOLO('modelterbaik.pt')  # Ensure the model file is available and accessible



    return model

if __name__ == "__main__":
    model = load_yolov8_model()
    print("YOLOv8 model loaded successfully.")
