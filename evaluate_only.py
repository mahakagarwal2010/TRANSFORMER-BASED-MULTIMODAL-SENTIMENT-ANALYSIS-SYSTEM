import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from multimodal_mental_health import MultimodalModel, MultimodalDataset, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_dir = r"C:\Users\MAHAK AGARWAL\multimodal system\crema\CREMA-D\AudioWAV"
video_dir = r"C:\Users\MAHAK AGARWAL\multimodal system\crema\CREMA-D\VideoFlash"

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

train_audio, test_audio, train_video, test_video, train_labels, test_labels = train_test_split(
    audio_paths,
    video_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

full_test = MultimodalDataset(test_audio, test_video, test_labels, max_frames=12, audio_seconds=5)

subset_size = 400
subset_size = min(subset_size, len(full_test))
test_dataset = Subset(full_test, list(range(subset_size)))

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

model = MultimodalModel(freeze_audio=True, freeze_vision=True).to(device)
model.load_state_dict(torch.load("multimodal_crema_model.pth", map_location=device))
model.eval()

print(f"Evaluating on subset: {subset_size} samples")
evaluate_model(model, test_loader)
