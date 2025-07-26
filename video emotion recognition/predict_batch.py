import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import Counter

label2idx = {
    "neutral": 0, "calm": 1, "happy": 2, "sad": 3,
    "angry": 4, "fearful": 5, "disgust": 6, "surprised": 7
}
idx2label = {v: k for k, v in label2idx.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(label2idx))
model.load_state_dict(torch.load("resnet18_ravdess.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

video_folder = "deneme video"
video_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".mp4")])

results = []

for video_name in video_files:
    video_path = os.path.join(video_folder, video_name)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = min(10, frame_count)
    frame_indices = [int(i * frame_count / sample_count) for i in range(sample_count)]

    predictions = []
    frame_id = 0
    success = True

    while success and frame_id <= max(frame_indices):
        success, frame = cap.read()
        if frame_id in frame_indices and success:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                _, pred = torch.max(output, 1)
                predictions.append(pred.item())
        frame_id += 1

    cap.release()

    if predictions:
        majority = Counter(predictions).most_common(1)[0][0]
        final_label = idx2label[majority]
    else:
        final_label = "yok (frame alınamadı)"

    results.append((video_name, final_label))

for video, label in results:
    print(f"{video:<20} → {label}")
