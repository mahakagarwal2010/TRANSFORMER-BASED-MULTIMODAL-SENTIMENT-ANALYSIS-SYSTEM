import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

video_dir = r"C:\Users\MAHAK AGARWAL\multimodal system\crema\CREMA-D\VideoFlash"

frame_counts = []
durations = []
brightness_values = []

for file in os.listdir(video_dir):
    if file.endswith(".flv"):
        path = os.path.join(video_dir, file)
        cap = cv2.VideoCapture(path)

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps > 0:
            duration = frames / fps
        else:
            duration = 0

        frame_counts.append(frames)
        durations.append(duration)

        # brightness check
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(np.mean(gray))

        cap.release()

# 📊 Frame Count
plt.figure()
plt.hist(frame_counts, bins=20)
plt.title("Frame Count Distribution")
plt.xlabel("Frames")
plt.ylabel("Frequency")
plt.show()

# ⏱ Duration
plt.figure()
plt.hist(durations, bins=20)
plt.title("Video Duration Distribution")
plt.xlabel("Seconds")
plt.ylabel("Frequency")
plt.show()

# 💡 Brightness
plt.figure()
plt.hist(brightness_values, bins=20)
plt.title("Frame Brightness Distribution")
plt.xlabel("Brightness")
plt.ylabel("Frequency")
plt.show()

# 🎥 Sample Frames
sample_video = os.listdir(video_dir)[0]
cap = cv2.VideoCapture(os.path.join(video_dir, sample_video))

plt.figure(figsize=(10,4))

for i in range(5):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.subplot(1,5,i+1)
    plt.imshow(frame)
    plt.axis("off")

plt.suptitle("Sample Video Frames")
plt.show()

cap.release()