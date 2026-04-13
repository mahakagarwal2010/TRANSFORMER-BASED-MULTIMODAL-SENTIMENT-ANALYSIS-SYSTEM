import os
import librosa
import cv2
import numpy as np
import matplotlib.pyplot as plt

audio_dir = r"C:\Users\MAHAK AGARWAL\multimodal system\crema\CREMA-D\AudioWAV"
video_dir = r"C:\Users\MAHAK AGARWAL\multimodal system\crema\CREMA-D\VideoFlash"

audio_energy = []
video_motion = []

files = os.listdir(audio_dir)[:50]  # sample subset

for file in files:
    if not file.endswith(".wav"):
        continue

    parts = file.split("_")
    if len(parts) < 3:
        continue

    audio_path = os.path.join(audio_dir, file)
    video_path = os.path.join(video_dir, os.path.splitext(file)[0] + ".flv")

    if not os.path.exists(video_path):
        continue

    # 🔊 Audio Energy
    audio, sr = librosa.load(audio_path, sr=16000)
    energy = np.sum(audio**2)
    audio_energy.append(energy)

    # 🎥 Video Motion
    cap = cv2.VideoCapture(video_path)
    prev = None
    motion = []

    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev is not None:
            diff = cv2.absdiff(prev, gray)
            motion.append(np.mean(diff))

        prev = gray

    cap.release()

    if len(motion) > 0:
        video_motion.append(np.mean(motion))

# 📊 Scatter plot
plt.figure()
plt.scatter(audio_energy[:len(video_motion)], video_motion)
plt.xlabel("Audio Energy")
plt.ylabel("Video Motion")
plt.title("Audio vs Video Correlation")
plt.show()