import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from PIL import Image
import os
from transformers import DeformableDetrForObjectDetection

# Load model
model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define paths
data_root = r"D:\Shreyan\Development\Train\Data\cat_detr"
train_images = os.path.join(data_root, r"train\images")
train_annotations = os.path.join(data_root, r"train\annotations\instances_train.json")

# Transform (Resizing is mandatory for DETR)
transform = T.Compose([
    T.Resize((800, 800)),  
    T.ToTensor()
])

# Custom Dataset Wrapper
class DetrCocoDataset(CocoDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img = transform(img)

        # Convert bbox format from (x, y, w, h) → (x_min, y_min, x_max, y_max)
        w, h = img.shape[1], img.shape[2]
        boxes = torch.tensor([obj["bbox"] for obj in target], dtype=torch.float32)
        if boxes.numel() == 0:  # Handle empty annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x_max = x + w
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y_max = y + h
            boxes[:, [0, 2]] /= w  # Normalize x_min, x_max
            boxes[:, [1, 3]] /= h  # Normalize y_min, y_max

        labels = torch.tensor([obj["category_id"] for obj in target], dtype=torch.int64)

        target_dict = {
            "boxes": boxes, 
            "labels": labels,
            "class_labels": labels  # ✅ Fix: Add class_labels key
        }
        return img, target_dict


# Load dataset
train_dataset = DetrCocoDataset(root=train_images, annFile=train_annotations)

# Custom collate function
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)  # Convert to Tensor
    return images, list(targets)  # Targets must remain list of dicts

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images = images.to(device)  # Move images to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move targets to device

        optimizer.zero_grad()
        
        # ✅ Fix: Pass 'labels' instead of 'class_labels'
        outputs = model(images, labels=targets)  

        loss = outputs.loss  # Hugging Face API
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Save fine-tuned model
torch.save(model.state_dict(), "deformable_detr_finetuned.pth")

# Inference on a test image
model.eval()
test_image_path = r"D:\Shreyan\Development\Train\Data\cat_cocoformat\test\images\test.jpg"
image = Image.open(test_image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)  # Ensure correct shape

with torch.no_grad():
    outputs = model(image)

print(outputs)  # View predicted boxes & labels
