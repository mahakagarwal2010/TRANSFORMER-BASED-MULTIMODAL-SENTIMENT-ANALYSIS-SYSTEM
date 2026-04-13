import time
from collections import deque
import cv2
import numpy as np
import torch
import sounddevice as sd
from multimodal_mental_health import MultimodalModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_SECONDS = 3
SR = 16000
AUDIO_PAD_SECONDS = 5
MAX_FRAMES = 12
MODEL_PATH = "multimodal_crema_model.pth"

THRESH = 0.4
SMOOTH_WINDOW = 3
probs_window = deque(maxlen=SMOOTH_WINDOW)


total_preds = 0
distress_count = 0
healthy_count = 0
all_confidences = []

def record_audio(seconds=AUDIO_SECONDS, sr=SR):
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()

def capture_frames(cap, max_frames=MAX_FRAMES, seconds=AUDIO_SECONDS):
    frames = []
    delay = seconds / max_frames if max_frames > 0 else 0

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)
        time.sleep(delay)

    while len(frames) < max_frames:
        frames.append(np.zeros((3, 224, 224), dtype=np.float32))

    return np.array(frames, dtype=np.float32)

def preprocess_audio(audio):
    max_len = int(SR * AUDIO_PAD_SECONDS)
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]
    return torch.tensor(audio, dtype=torch.float32)

# Load model
model = MultimodalModel(freeze_audio=True, freeze_vision=True).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state, strict=True)
model.eval()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

print("Realtime demo running. Press Q to quit (Ctrl+C also works).")

try:
    while True:
        audio = record_audio()
        frames = capture_frames(cap)

        audio_t = preprocess_audio(audio).unsqueeze(0).to(device)
        video_t = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(audio_t, video_t)
            prob = torch.sigmoid(logits).item()

        probs_window.append(prob)
        prob_smooth = float(np.median(np.array(probs_window)))

        pred = "DISTRESS" if prob_smooth > THRESH else "HEALTHY"

       
        total_preds += 1
        all_confidences.append(prob_smooth)

        if pred == "DISTRESS":
            distress_count += 1
        else:
            healthy_count += 1

        avg_conf = np.mean(all_confidences)
        distress_ratio = distress_count / total_preds if total_preds > 0 else 0

        # Display
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, f"{pred} conf={prob_smooth:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"AvgConf={avg_conf:.2f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, f"Distress%={distress_ratio*100:.1f}%", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Realtime Mental State Demo", frame)

        print(f"Prediction: {pred} | Conf: {prob_smooth:.2f} | Avg: {avg_conf:.2f}")

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()

   
    print("\n===== SESSION SUMMARY =====")
    print(f"Total Predictions : {total_preds}")
    print(f"Healthy Count     : {healthy_count}")
    print(f"Distress Count    : {distress_count}")
    print(f"Distress %        : {(distress_count/total_preds)*100:.2f}%")
    print(f"Average Confidence: {np.mean(all_confidences):.2f}")