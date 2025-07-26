import os
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification


MODEL_DIR = "model"


processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
model = AutoModelForAudioClassification.from_pretrained(MODEL_DIR)
model.eval()


AUDIO_DIR = "audio test video"
files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]


id2label = model.config.id2label

for file in sorted(files):
    path = os.path.join(AUDIO_DIR, file)
    speech, sr = torchaudio.load(path)

    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)

    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        speech = resampler(speech)
        sr = 16000

    inputs = processor(speech.squeeze(), sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        label = id2label[predicted_id]

    print(f"{file} → Tahmin: {label}")
