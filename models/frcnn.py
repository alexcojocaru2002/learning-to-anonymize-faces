import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------- DALY Action Classes ----------
# Index 0 is background by convention in Faster R-CNN
action_classes = {
    1: "drinking",
    2: "phoning",
    3: "cleaning floor",
    4: "cleaning windows",
    5: "ironing",
    6: "folding textile",
    7: "playing harmonica",
    8: "taking photos",
    9: "vacuum cleaning",
    10: "watching TV"
}

# ---------- Load Pretrained Model ----------
device = torch.device("mps" if torch.mps.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_action_classes = len(action_classes)
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_action_classes + 1)
model.to(device)
model.eval()

# ---------- Load and Transform DALY Frame ----------
transform = transforms.Compose([  # Resize to DALY paper specs
    transforms.ToTensor()
])

# Load your video frame image here
image_path = "drink.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).to(device)

# ---------- Run Inference ----------
with torch.no_grad():
    prediction = model([image_tensor])[0]

# ---------- Print Predictions ----------
print("\n--- Predictions ---")
for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
    # if score > 0.5:
    print(f"Action: {label.item()}, "
            f"Score: {score:.2f}, Box: {box.cpu().numpy()}")

# ---------- Visualize ----------
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
    if score > 0.5:
        x1, y1, x2, y2 = box.cpu().numpy()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10,
                f"{label.item()}: {score:.2f}",
                color='red', fontsize=10, backgroundcolor='white')

plt.axis('off')
plt.tight_layout()
plt.show() 