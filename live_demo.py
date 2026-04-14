"""
Live ASL fingerspelling recognition demo using webcam.

Opens the webcam, detects hand landmarks in real-time using MediaPipe Tasks API,
feeds sequences through the trained LSTM model, and displays predicted
letters with accumulated text.

Controls:
    SPACE     - Add a space to the text
    BACKSPACE - Delete last character
    C         - Clear all text
    ESC       - Quit

Usage:
    python live_demo.py
"""

import os
import json
import time
import collections
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
)
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import torch

# ──────────────────── Import model class from train_model ──────────────
from train_model import ASLLSTM

# ──────────────────────────── Configuration ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "asl_lstm_model.pth")
HAND_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "hand_landmarker.task")
CONFIDENCE_THRESHOLD = 0.6   # Minimum confidence to display prediction
COMMIT_THRESHOLD = 0.7       # Minimum confidence to commit a letter
STABILITY_FRAMES = 15        # Consecutive frames needed to commit a letter
COOLDOWN_FRAMES = 10         # Frames to wait after committing before next commit
SEQ_LEN = 10                 # Must match training sequence length

# ──────────────────────────── Colors (BGR) ─────────────────────────────
COLOR_BG_PANEL = (40, 40, 40)
COLOR_TEXT_PRIMARY = (255, 255, 255)
COLOR_TEXT_SECONDARY = (180, 180, 180)
COLOR_ACCENT = (0, 200, 120)
COLOR_WARNING = (0, 100, 255)
COLOR_HAND_CONNECTIONS = (50, 205, 50)
COLOR_HAND_LANDMARKS = (0, 140, 255)
COLOR_CONFIDENCE_HIGH = (0, 200, 120)
COLOR_CONFIDENCE_MED = (0, 200, 255)
COLOR_CONFIDENCE_LOW = (0, 100, 255)

# MediaPipe hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


def load_model():
    """Load the trained LSTM model and label mappings."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run train_model.py first to train the model."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model = ASLLSTM(
        input_size=checkpoint["input_size"],
        num_classes=checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx_to_label = checkpoint["idx_to_label"]
    # Convert string keys back to int
    idx_to_label = {int(k): v for k, v in idx_to_label.items()}

    return model, idx_to_label, device


def extract_landmarks(hand_landmarks) -> np.ndarray:
    """Extract and normalize 63 features from MediaPipe hand landmarks.

    Normalizes by centering on wrist and scaling by max 2D distance,
    matching the preprocessing used during training.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)

    # Normalize: center on wrist
    wrist = coords[0].copy()
    coords -= wrist

    # Normalize: scale by max 2D distance from wrist
    max_dist = np.sqrt((coords[:, :2] ** 2).sum(axis=1)).max()
    if max_dist > 1e-6:
        coords /= max_dist

    return coords.flatten()


def get_confidence_color(confidence: float) -> tuple:
    """Return BGR color based on confidence level."""
    if confidence >= 0.8:
        return COLOR_CONFIDENCE_HIGH
    elif confidence >= 0.5:
        return COLOR_CONFIDENCE_MED
    else:
        return COLOR_CONFIDENCE_LOW


def draw_hand_on_frame(frame, hand_landmarks, frame_w, frame_h):
    """Draw hand landmarks and connections directly on the frame."""
    points = []
    for lm in hand_landmarks:
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        points.append((x, y))

    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx],
                     COLOR_HAND_CONNECTIONS, 2)

    # Draw landmarks
    for (x, y) in points:
        cv2.circle(frame, (x, y), 4, COLOR_HAND_LANDMARKS, -1)
        cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)


def draw_info_panel(frame, predicted_char, confidence, accumulated_text,
                    hand_detected, fps):
    """Draw the information overlay panel on the frame."""
    h, w = frame.shape[:2]
    panel_height = 140

    # Semi-transparent bottom panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_height), (w, h), COLOR_BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Divider line
    cv2.line(frame, (0, h - panel_height), (w, h - panel_height), COLOR_ACCENT, 2)

    y_base = h - panel_height + 30

    # Current prediction
    if hand_detected and predicted_char and confidence >= CONFIDENCE_THRESHOLD:
        conf_color = get_confidence_color(confidence)
        pred_display = predicted_char.upper() if len(predicted_char) == 1 else predicted_char

        cv2.putText(frame, f"Sign: {pred_display}", (20, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, conf_color, 2)

        # Confidence bar
        bar_x, bar_y = 250, y_base - 18
        bar_w, bar_h = 150, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), -1)
        filled_w = int(bar_w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h),
                      conf_color, -1)
        cv2.putText(frame, f"{confidence:.0%}", (bar_x + bar_w + 10, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_SECONDARY, 1)
    else:
        status = "No hand detected" if not hand_detected else "Low confidence"
        cv2.putText(frame, status, (20, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WARNING, 2)

    # Accumulated text
    cv2.putText(frame, "Text:", (20, y_base + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_SECONDARY, 1)
    # Truncate display text if too long
    display_text = accumulated_text[-40:] if len(accumulated_text) > 40 else accumulated_text
    if len(accumulated_text) > 40:
        display_text = "..." + display_text
    cv2.putText(frame, display_text + "_", (90, y_base + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT_PRIMARY, 2)

    # Controls help
    cv2.putText(frame, "SPACE:space | BACKSPACE:del | C:clear | ESC:quit",
                (20, y_base + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_SECONDARY, 1)

    # FPS counter (top-right)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ACCENT, 2)

    # Hand detection indicator (top-left)
    indicator_color = COLOR_ACCENT if hand_detected else COLOR_WARNING
    cv2.circle(frame, (25, 25), 10, indicator_color, -1)
    cv2.putText(frame, "Hand" if hand_detected else "No Hand", (45, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, indicator_color, 2)


def main():
    """Run the live ASL recognition demo."""
    print("Loading LSTM model...")
    model, idx_to_label, device = load_model()
    print(f"Model loaded. Classes: {len(idx_to_label)}")
    print(f"Device: {device}")

    # MediaPipe HandLandmarker setup (Tasks API)
    print("Loading MediaPipe HandLandmarker...")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)
    print("HandLandmarker ready.")

    # State
    landmark_buffer = collections.deque(maxlen=SEQ_LEN)
    accumulated_text = ""
    last_predicted = None
    stability_counter = 0
    cooldown_counter = 0
    fps = 0
    prev_time = time.time()
    frame_timestamp_ms = 0

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Make sure a camera is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 50)
    print("ASL Live Recognition Started!")
    print("Show ASL hand signs to the camera.")
    print("Controls: SPACE=space, BACKSPACE=delete, C=clear, ESC=quit")
    print("=" * 50 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 0.001)
            prev_time = current_time

            # Calculate timestamp for VIDEO mode
            frame_timestamp_ms += int(1000 / 30)  # Assume ~30fps

            # Process with MediaPipe Tasks API
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            hand_detected = False
            predicted_char = None
            confidence = 0.0

            if result.hand_landmarks:
                hand_detected = True
                hand_landmarks = result.hand_landmarks[0]

                # Draw landmarks manually
                draw_hand_on_frame(frame, hand_landmarks, frame_w, frame_h)

                # Extract and buffer landmarks
                landmarks = extract_landmarks(hand_landmarks)
                landmark_buffer.append(landmarks)

                # Predict when buffer is full
                if len(landmark_buffer) == SEQ_LEN:
                    sequence = np.array(list(landmark_buffer), dtype=np.float32)
                    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(sequence_tensor)
                        probs = torch.softmax(output, dim=1)
                        conf, pred_idx = probs.max(1)
                        confidence = conf.item()
                        pred_label = idx_to_label[pred_idx.item()]
                        predicted_char = pred_label

                    # Stability logic for text commitment
                    if cooldown_counter > 0:
                        cooldown_counter -= 1
                    elif confidence >= COMMIT_THRESHOLD:
                        if predicted_char == last_predicted:
                            stability_counter += 1
                        else:
                            stability_counter = 1
                            last_predicted = predicted_char

                        if stability_counter >= STABILITY_FRAMES:
                            accumulated_text += predicted_char
                            print(f"Committed: '{predicted_char}' → Text: '{accumulated_text}'")
                            stability_counter = 0
                            cooldown_counter = COOLDOWN_FRAMES
                    else:
                        stability_counter = 0
            else:
                # No hand detected
                stability_counter = 0

            # Draw UI overlay
            draw_info_panel(frame, predicted_char, confidence, accumulated_text,
                           hand_detected, fps)

            # Display
            cv2.imshow("ASL to Text - Live Demo", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(" "):  # SPACE
                accumulated_text += " "
                print(f"Space added → Text: '{accumulated_text}'")
            elif key == 8 or key == 127:  # BACKSPACE
                accumulated_text = accumulated_text[:-1]
                print(f"Backspace → Text: '{accumulated_text}'")
            elif key == ord("c") or key == ord("C"):
                accumulated_text = ""
                print("Text cleared")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print(f"\nFinal text: '{accumulated_text}'")


if __name__ == "__main__":
    main()
