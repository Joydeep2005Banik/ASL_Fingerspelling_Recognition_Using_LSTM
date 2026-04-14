"""
Train an LSTM model for ASL fingerspelling recognition.

Uses preprocessed MediaPipe hand landmarks and creates synthetic temporal
sequences to train an LSTM classifier. Since the dataset contains static
images, each sample is augmented into a sequence of SEQ_LEN frames with
small random jitter to simulate temporal variation.

Architecture:
    Input(SEQ_LEN, 63) → LSTM(128) → Dropout → LSTM(64) → Dropout
    → Dense(64, ReLU) → Dense(num_classes, Softmax)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────── Configuration ────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LEN = 10          # Number of frames in each synthetic sequence
JITTER_STD = 0.02     # Standard deviation of landmark jitter
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10         # Early stopping patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────── Dataset ──────────────────────────────────
class ASLSequenceDataset(Dataset):
    """
    Creates synthetic temporal sequences from static landmark samples.

    For each sample, generates SEQ_LEN copies with random jitter added
    to simulate the natural variation seen in consecutive video frames.
    """

    def __init__(self, landmarks: np.ndarray, labels: np.ndarray,
                 seq_len: int = SEQ_LEN, jitter_std: float = JITTER_STD,
                 augment: bool = True):
        self.landmarks = landmarks  # (N, 63)
        self.labels = labels        # (N,)
        self.seq_len = seq_len
        self.jitter_std = jitter_std
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        base = self.landmarks[idx]  # (63,)
        label = self.labels[idx]

        # Create a sequence of seq_len frames
        if self.augment:
            jitter = np.random.randn(self.seq_len, 63).astype(np.float32) * self.jitter_std
            sequence = np.tile(base, (self.seq_len, 1)) + jitter
        else:
            sequence = np.tile(base, (self.seq_len, 1))

        return (
            torch.tensor(sequence, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


# ──────────────────────────── Model ────────────────────────────────────
class ASLLSTM(nn.Module):
    """
    LSTM-based classifier for ASL fingerspelling recognition.

    Architecture:
        LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.3)
        → Dense(64, ReLU) → Dense(num_classes)
    """

    def __init__(self, input_size: int = 63, hidden1: int = 128,
                 hidden2: int = 64, num_classes: int = 36,
                 dropout: float = 0.3):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden1,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden1,
            hidden_size=hidden2,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, 63)
        out, _ = self.lstm1(x)       # (batch, seq_len, 128)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)     # (batch, seq_len, 64)
        out = self.dropout2(out)

        # Take only the last time step
        out = out[:, -1, :]          # (batch, 64)

        out = self.relu(self.fc1(out))  # (batch, 64)
        out = self.fc2(out)             # (batch, num_classes)
        return out


# ──────────────────────────── Training ─────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels in loader:
        sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    """Evaluate model and return loss, accuracy, and predictions."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Save training history plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label="Train Loss", linewidth=2)
    ax1.plot(val_losses, label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label="Train Acc", linewidth=2)
    ax2.plot(val_accs, label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history saved to {save_path}")


def main():
    """Main training pipeline."""
    # ─── Load data ───
    print("Loading preprocessed landmarks...")
    landmarks = np.load(os.path.join(DATA_DIR, "landmarks.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))

    with open(os.path.join(DATA_DIR, "label_map.json")) as f:
        label_map = json.load(f)

    num_classes = len(label_map)
    idx_to_label = {v: k for k, v in label_map.items()}

    print(f"Loaded {len(landmarks)} samples, {num_classes} classes")
    print(f"Landmarks shape: {landmarks.shape}")

    # ─── Train/test split ───
    X_train, X_test, y_train, y_test = train_test_split(
        landmarks, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ─── Create dataloaders ───
    train_dataset = ASLSequenceDataset(X_train, y_train, augment=True)
    test_dataset = ASLSequenceDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ─── Initialize model ───
    model = ASLLSTM(
        input_size=63,
        hidden1=128,
        hidden2=64,
        num_classes=num_classes,
        dropout=0.3,
    ).to(DEVICE)

    print(f"\nModel architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Device: {DEVICE}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ─── Training loop ───
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(MODEL_DIR, "asl_lstm_model.pth")

    print("Starting training...")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.1%} | {val_loss:>10.4f} | {val_acc:>8.1%}")

        # Early stopping & checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_map": label_map,
                "idx_to_label": idx_to_label,
                "num_classes": num_classes,
                "seq_len": SEQ_LEN,
                "input_size": 63,
            }, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # ─── Plot training history ───
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(MODEL_DIR, "training_history.png"),
    )

    # ─── Final evaluation ───
    print("\n" + "=" * 60)
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, final_acc, preds, true_labels = evaluate(model, test_loader, criterion)
    print(f"Best Validation Accuracy: {final_acc:.1%}")

    all_label_indices = list(range(num_classes))
    class_names = [idx_to_label[i] for i in all_label_indices]
    print("\nClassification Report:")
    print(classification_report(
        true_labels, preds,
        labels=all_label_indices,
        target_names=class_names,
        zero_division=0,
    ))

    print(f"\nModel saved to: {best_model_path}")


if __name__ == "__main__":
    main()
