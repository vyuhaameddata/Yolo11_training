from ultralytics import YOLO
model = YOLO(r"yolo11s.pt")  # Load YOLOv11 model
print(model)

import torch
print(torch.cuda.is_available())  # Should print True
print(torch.version.cuda)  # Should match your installed CUDA version (e.g., 11.8)
print(torch.backends.cudnn.version())  # Should return a valid cuDNN version