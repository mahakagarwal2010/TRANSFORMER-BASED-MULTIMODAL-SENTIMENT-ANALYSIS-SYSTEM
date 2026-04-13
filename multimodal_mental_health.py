import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import Wav2Vec2Model
import librosa
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultimodalDataset(Dataset):
    def __init__(self, audio_paths, video_paths, labels, max_frames=16, audio_seconds=5):
        self.audio_paths = audio_paths
        self.video_paths = video_paths
        self.labels = labels
        self.max_frames = max_frames
        self.audio_seconds = audio_seconds

    def load_audio(self, path):
        audio, sr = librosa.load(path, sr=16000)
        max_len = int(self.audio_seconds * sr)
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        else:
            audio = audio[:max_len]
        return torch.tensor(audio, dtype=torch.float32)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        if total and total > 0:
            idxs = np.linspace(0, max(total - 1, 0), self.max_frames).astype(int)
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame.astype(np.float32) / 255.0
                frame = np.transpose(frame, (2, 0, 1))
                frames.append(frame)
        else:
            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame.astype(np.float32) / 255.0
                frame = np.transpose(frame, (2, 0, 1))
                frames.append(frame)

        cap.release()

        while len(frames) < self.max_frames:
            frames.append(np.zeros((3, 224, 224), dtype=np.float32))

        return torch.tensor(np.array(frames), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = self.load_audio(self.audio_paths[idx])
        video = self.load_video(self.video_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return audio, video, label


class AudioEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        if freeze:
            for p in self.wav2vec.parameters():
                p.requires_grad = False
        self.projection = nn.Linear(768, 256)

    def forward(self, input_values):
        outputs = self.wav2vec(input_values).last_hidden_state
        pooled = torch.mean(outputs, dim=1)
        return self.projection(pooled)


class FaceEncoder(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=2
        )
        self.projection = nn.Linear(512, 256)

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        features = self.feature_extractor(frames)
        features = features.view(B, T, 512)
        temporal_features = self.temporal_transformer(features)
        pooled = torch.mean(temporal_features, dim=1)
        return self.projection(pooled)


class MultimodalModel(nn.Module):
    def __init__(self, freeze_audio=True, freeze_vision=True):
        super().__init__()
        self.audio_encoder = AudioEncoder(freeze=freeze_audio)
        self.face_encoder = FaceEncoder(freeze_backbone=freeze_vision)
        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, audio, frames):
        audio_embed = self.audio_encoder(audio).unsqueeze(1)
        face_embed = self.face_encoder(frames).unsqueeze(1)
        fused, _ = self.cross_attention(audio_embed, face_embed, face_embed)
        fused = fused.squeeze(1)
        return self.classifier(fused)


def train_model(model, loader, epochs=2, lr=1e-4, grad_clip=1.0):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print("Training started...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for step, (audio, video, label) in enumerate(loader):
            if step % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {step}/{len(loader)}")

            audio = audio.to(device, non_blocking=True)
            video = video.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(audio, video)
                loss = criterion(outputs, label)

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")


def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for audio, video, label in loader:
            audio = audio.to(device, non_blocking=True)
            video = video.to(device, non_blocking=True)
            outputs = model(audio, video)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(label.numpy().astype(int).tolist())

    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("F1 Score:", f1_score(all_labels, all_preds))
    try:
        print("ROC-AUC:", roc_auc_score(all_labels, all_preds))
    except Exception:
        print("ROC-AUC: not computable")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

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
            video_name = os.path.splitext(file)[0] + ".flv"
            video_file = os.path.join(video_dir, video_name)
            if not os.path.exists(video_file):
                continue
            label = 1 if emotion in ["SAD", "ANG", "FEA"] else 0
            audio_paths.append(audio_file)
            video_paths.append(video_file)
            labels.append(label)

    print("Total paired samples:", len(labels))

    if len(labels) < 10:
        raise RuntimeError("Too few paired samples found")

    test_vid = None
    for vp in video_paths[:50]:
        cap = cv2.VideoCapture(vp)
        ok = cap.isOpened()
        ret, _ = cap.read() if ok else (False, None)
        cap.release()
        if ok and ret:
            test_vid = vp
            break

    print("Video test file:", test_vid)
    print("Video readable:", test_vid is not None)

    if test_vid is None:
        raise RuntimeError("Videos not readable")

    train_audio, test_audio, train_video, test_video, train_labels, test_labels = train_test_split(
        audio_paths,
        video_paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    train_dataset = MultimodalDataset(train_audio, train_video, train_labels, max_frames=12, audio_seconds=5)
    test_dataset = MultimodalDataset(test_audio, test_video, test_labels, max_frames=12, audio_seconds=5)

    train_limit = min(1500, len(train_dataset))
    test_limit = min(400, len(test_dataset))

    train_dataset = Subset(train_dataset, list(range(train_limit)))
    test_dataset = Subset(test_dataset, list(range(test_limit)))

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=pin)

    model = MultimodalModel(freeze_audio=True, freeze_vision=True).to(device)

    train_model(model, train_loader, epochs=2, lr=1e-4)
    evaluate_model(model, test_loader)

    torch.save(model.state_dict(), "multimodal_crema_model.pth")
    print("Model saved")
