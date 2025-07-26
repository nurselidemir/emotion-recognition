#  Video-Based Emotion Recognition with ResNet18

This project performs emotion recognition from short video clips using a ResNet18 model trained on the RAVDESS dataset.

The model analyzes a video by extracting 10 evenly spaced frames and predicting the emotion expressed in each. A majority voting strategy is used to determine the final label.

---

## Features

- Input: short `.mp4` videos (~3 seconds)
- Extracts 10 frames per video
- ResNet18 model trained on RAVDESS emotion expressions
- Majority voting over frames to decide emotion
- Includes test videos under `deneme video/`

---


##  Model Training Info

- **Dataset:** [RAVDESS Emotional Speech Video Dataset (Kaggle)](https://www.kaggle.com/datasets/adrivg/ravdess-emotional-speech-video)
- **Trained on:** Extracted video frames from RAVDESS  
- **Architecture:** ResNet18 (final FC layer adapted to 8 emotion classes)  
- **Training strategy:** Data augmentation + class weighting


## Emotion Classes

The model predicts one of the following 8 emotion classes:

- `neutral`
- `calm`
- `happy`
- `sad`
- `angry`
- `fearful`
- `disgust`
- `surprised`

---

## Folder Structure

- `deneme video`: Test videos (.mp4)
- `resnet18_ravdess.pth `: Trained model weights
- `predict_batch.py`: Main batch prediction script
- `README.md`: Project documentation


## 🚀 Getting Started

### Clone the repo
```bash
git clone https://github.com/nurselidemir/video-emotion-recognition.git
cd video-emotion-recognition
```

### Install dependencies
```bash
pip install torch torchvision opencv-python pillow
```

### Run batch prediction
```bash
python predict_batch.py
```

---

#### Example Output

```bash
angry.mp4            → angry
happy.mp4            → happy
neutral.mp4          → neutral
sad.mp4              → sad
surprised.mp4        → surprised
```
---

