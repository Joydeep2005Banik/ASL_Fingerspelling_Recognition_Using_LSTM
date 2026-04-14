"""
Extract hand landmarks from ASL dataset images using MediaPipe Tasks API.

Processes all images in asl_dataset/, extracts 21 hand landmarks (63 features)
per image, and saves the results for LSTM training.

Output:
    data/landmarks.npy  - Shape (N, 63) array of landmark features
    data/labels.npy     - Shape (N,) array of integer labels
    data/label_map.json - Mapping from class name to integer label
"""

import os
import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
)
from mediapipe.tasks.python import BaseOptions

# ──────────────────────────── Configuration ────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asl_dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}

# ──────────────────────────── Setup ────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create HandLandmarker with the Tasks API
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)


def extract_landmarks_from_image(image_path: str) -> np.ndarray | None:
    """
    Extract 21 hand landmarks from an image.

    Returns:
        numpy array of shape (63,) with [x, y, z] for each of 21 landmarks,
        or None if no hand is detected.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image from numpy array
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Detect hand landmarks
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    # Extract landmarks from the first detected hand
    hand_landmarks = result.hand_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)

    # Normalize: center on wrist
    wrist = coords[0].copy()
    coords -= wrist

    # Normalize: scale by max 2D distance from wrist
    max_dist = np.sqrt((coords[:, :2] ** 2).sum(axis=1)).max()
    if max_dist > 1e-6:
        coords /= max_dist

    return coords.flatten()


def main():
    """Process all images in the dataset and save extracted landmarks."""
    # Build sorted list of class directories (a-z, 0-9)
    class_dirs = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d)) and d != "asl_dataset"
    ])

    print(f"Found {len(class_dirs)} classes: {class_dirs}")

    label_map = {name: idx for idx, name in enumerate(class_dirs)}
    all_landmarks = []
    all_labels = []
    skipped = 0
    processed = 0

    for class_name in class_dirs:
        class_path = os.path.join(DATASET_DIR, class_name)
        image_files = sorted([
            f for f in os.listdir(class_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])

        class_count = 0
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            landmarks = extract_landmarks_from_image(img_path)

            if landmarks is not None:
                all_landmarks.append(landmarks)
                all_labels.append(label_map[class_name])
                class_count += 1
                processed += 1
            else:
                skipped += 1

        print(f"  Class '{class_name}': {class_count}/{len(image_files)} images processed")

    # Convert to numpy arrays
    landmarks_array = np.array(all_landmarks, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    # Save results
    np.save(os.path.join(OUTPUT_DIR, "landmarks.npy"), landmarks_array)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels_array)

    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Extraction complete!")
    print(f"  Total processed: {processed}")
    print(f"  Total skipped (no hand detected): {skipped}")
    print(f"  Landmarks shape: {landmarks_array.shape}")
    print(f"  Labels shape: {labels_array.shape}")
    print(f"  Saved to: {OUTPUT_DIR}/")
    print(f"  Label map: {label_map}")

    landmarker.close()


if __name__ == "__main__":
    main()
