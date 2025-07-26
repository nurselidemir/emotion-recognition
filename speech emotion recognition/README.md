#  Speech-Based Emotion Recognition with Wav2Vec2.0

This project performs emotion recognition from short speech audio clips using a fine-tuned Wav2Vec2.0 model on the RAVDESS dataset.

The model analyzes a `.wav` file and classifies the underlying emotion expressed by the speaker.

---

## Features

- Input: short `.wav` audio files (~3 seconds)
- Pretrained `Wav2Vec2.0` model fine-tuned on emotional speech
- Predicts 1 of 8 emotion classes per audio clip
- Includes sample test audios under `audio test video/`

---

## Model Training Info

- **Dataset:** [RAVDESS Emotional Speech Audio Dataset (Kaggle)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- **Trained on:** Raw `.wav` audio files  
- **Architecture:** `superb/wav2vec2-base-superb-er` (HuggingFace)  
- **Fine-tuning:** Final classification head adapted to 8-class emotion task  
- **Framework:** HuggingFace Transformers, PyTorch

---

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

- `audio test video/`: Test `.wav` audio files
- `model/`: Trained model files (requires Git LFS)
- `inference.py`: Inference script for predicting emotions
- `superb-wav2vec2-base-superb-er.ipynb`: Notebook for training and evaluation
- `README.md`: Project documentation

---

##  Git LFS Required

This project uses Git Large File Storage (LFS) for the pretrained model file (`model.safetensors`).

To clone and use this project properly, make sure Git LFS is installed:

```bash
git lfs install
git clone https://github.com/nurselidemir/emotion-recognition.git
``` 

### Install dependencies
```bash
pip install torch torchaudio transformers soundfile
```

### Run  prediction
```bash
python inference.py
```

#### Example Output

```bash
happy.wav         → happy
sad2.wav          → sad
angry.wav         → angry
surprised.wav     → surprised
```