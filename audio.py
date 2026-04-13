import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

audio_dir = r"C:\Users\MAHAK AGARWAL\multimodal system\crema\CREMA-D\AudioWAV"

emotions = []
labels = []
lengths = []

for file in os.listdir(audio_dir):
    if file.endswith(".wav"):
        parts = file.split("_")
        if len(parts) < 3:
            continue
        
        emotion = parts[2].upper()
        emotions.append(emotion)

        label = 1 if emotion in ["SAD", "ANG", "FEA"] else 0
        labels.append(label)

        path = os.path.join(audio_dir, file)
        audio, sr = librosa.load(path, sr=16000)
        lengths.append(len(audio)/sr)

# 📊 Class Distribution
plt.figure()
sns.countplot(x=labels)
plt.title("Audio Class Distribution (0=Healthy, 1=Distress)")
plt.show()

# 🎭 Emotion Distribution
plt.figure()
sns.countplot(x=emotions)
plt.title("Audio Emotion Distribution")
plt.xticks(rotation=45)
plt.show()

# ⏱ Audio Length
plt.figure()
plt.hist(lengths, bins=20)
plt.title("Audio Length Distribution")
plt.show()

# 🎵 Waveform
sample = os.listdir(audio_dir)[0]
audio, sr = librosa.load(os.path.join(audio_dir, sample), sr=16000)

plt.figure()
plt.plot(audio)
plt.title("Audio Waveform")
plt.show()