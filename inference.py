import torch
import cv2
from torchvision.transforms import transforms
from model.divita import build_divita

# Load Pretrained Model
model = build_divita(inum_features=768, vnum_features=768, fusion='late', cam='tsfm')
model.load_state_dict(torch.load('epoch.ckpt'))
model.eval()

# Video Preprocessing
video_path = 'Oppenheimer.mp4'
cap = cv2.VideoCapture(video_path)
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Resize frame if needed
    frame = cv2.resize(frame, (224, 224))
    frames.append(frame)

cap.release()
video_data = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)  # Assuming HWC format

# Inference
with torch.no_grad():
    predictions = model(video_data)

# Process predictions as needed for your task