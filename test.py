import os
import time
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from multimodal_mental_health import MultimodalModel, MultimodalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# 1. DATA PATHS
# ==============================
audio_dir = r"C:\Users\MAHAK AGARWAL\multimodal system\crema\CREMA-D\AudioWAV"
video_dir = r"C:\Users\MAHAK AGARWAL\multimodal system\crema\CREMA-D\VideoFlash"
model_path = "multimodal_crema_model.pth"

# ==============================
# 2. LOAD PAIRED FILES
# ==============================
audio_paths = []
video_paths = []
labels = []

for file in os.listdir(audio_dir):
    if file.lower().endswith(".wav"):
        parts = file.split("_")
        if len(parts) < 3:
            continue

        emotion = parts[2].upper()
        audio_file = os.path.join(audio_dir, file)
        video_file = os.path.join(video_dir, os.path.splitext(file)[0] + ".flv")

        if not os.path.exists(video_file):
            continue

        label = 1 if emotion in ["SAD", "ANG", "FEA"] else 0

        audio_paths.append(audio_file)
        video_paths.append(video_file)
        labels.append(label)

print(f"Total paired samples found: {len(labels)}")

if len(labels) == 0:
    raise RuntimeError("No paired audio-video samples found. Check dataset paths.")

# ==============================
# 3. TRAIN / TEST SPLIT
# ==============================
train_audio, test_audio, train_video, test_video, train_labels, test_labels = train_test_split(
    audio_paths,
    video_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"Training samples: {len(train_labels)}")
print(f"Testing samples : {len(test_labels)}")

# ==============================
# 4. CREATE TEST DATASET
# ==============================
full_test = MultimodalDataset(
    test_audio,
    test_video,
    test_labels,
    max_frames=12,
    audio_seconds=5
)

# Reduce for faster evaluation
subset_size = 50
subset_size = min(subset_size, len(full_test))
test_dataset = Subset(full_test, list(range(subset_size)))

print(f"Evaluating on subset: {subset_size} samples")

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

# ==============================
# 5. LOAD MODEL
# ==============================
model = MultimodalModel(freeze_audio=True, freeze_vision=True).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ==============================
# 6. EVALUATE MODEL
# ==============================
all_labels = []
all_preds = []
all_probs = []

start_time = time.time()

with torch.no_grad():
    for batch_idx, (audio, video, label) in enumerate(test_loader):
        print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")

        audio = audio.to(device)
        video = video.to(device)

        outputs = model(audio, video)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(label.numpy().astype(int).tolist())

end_time = time.time()

# ==============================
# 7. METRICS
# ==============================
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)

try:
    roc_auc = roc_auc_score(all_labels, all_probs)
except ValueError:
    roc_auc = None

cm = confusion_matrix(all_labels, all_preds)

# ==============================
# 8. PRINT RESULTS
# ==============================
print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy   : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")

if roc_auc is not None:
    print(f"ROC-AUC    : {roc_auc:.4f}")
else:
    print("ROC-AUC    : Not computable")

print("\nConfusion Matrix:")
print(cm)

print(f"\nEvaluation time: {end_time - start_time:.2f} seconds")
print("======================================")