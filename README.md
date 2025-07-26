#  Emotion Recognition Toolkit

This repository contains two separate emotion recognition modules:

1. **Video-Based Emotion Recognition** – using facial expressions in video clips  
2. **Speech-Based Emotion Recognition** – using tone and acoustic features in audio

Each module is independently trained and can be used separately or combined for multimodal analysis.

---

##  Modules

###  Video-Based Emotion Recognition

- **Model:** ResNet18
- **Dataset:** [RAVDESS Emotional Speech Video](https://www.kaggle.com/datasets/adrivg/ravdess-emotional-speech-video)
- **Inputs:** Short `.mp4` videos (~3 seconds)
- **Output:** Predicted emotion label from facial expression

📁 Folder: [`video emotion recognition/`](./video%20emotion%20recognition)

###  Speech-Based Emotion Recognition

- **Model:** Fine-tuned Wav2Vec2.0 (`superb/wav2vec2-base-superb-er`)
- **Dataset:** [RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- **Inputs:** Short `.wav` audio files
- **Output:** Predicted emotion label from vocal tone

📁 Folder: [`speech emotion recognition/`](./speech%20emotion%20recognition)

---

##  Emotion Classes

Both models are trained to predict one of the following 8 emotions:

- `neutral`
- `calm`
- `happy`
- `sad`
- `angry`
- `fearful`
- `disgust`
- `surprised`

---

##  Getting Started

### 1. Clone the Repository

```bash
git lfs install
git clone https://github.com/nurselidemir/emotion-recognition.git
```
### 2. Navigate into a module
```bash
cd "video emotion recognition"
# or
cd "speech emotion recognition"
```

## Requirements

- Python 3.8+
- torch, torchvision, torchaudio
- transformers
- opencv-python, soundfile, pillow
- Git LFS (for downloading model files)

