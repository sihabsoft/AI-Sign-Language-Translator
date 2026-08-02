"""
utils.py
--------
Shared helper functions for the AI Powered Sign Language Translator krishano.

Contains:
    - HandDetector : hand detection that works with BOTH the legacy
      MediaPipe "solutions" API (<= 0.10.14) and the new "tasks" API
      (>= 0.10.15, where mp.solutions was removed)
    - normalize_landmarks() : 21 landmarks -> normalized 63-value vector
    - dataset helpers used by collect_data.py and train_model.py
"""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

# ----------------------------------------------------------------------
# Configuration shared across the whole project
# ----------------------------------------------------------------------

DATASET_DIR = "dataset"
MODEL_PATH = os.path.join("models", "sign_model.pkl")

NUM_LANDMARKS = 21          # MediaPipe returns 21 points per hand
FEATURES_PER_LANDMARK = 3   # x, y, z
NUM_FEATURES = NUM_LANDMARKS * FEATURES_PER_LANDMARK   # 63

# Gesture label -> keyboard key used during data collection
GESTURES = {
    "hello":  "h",
    "thanks": "t",
    "yes":    "y",
    "no":     "n",
    "please": "p",
    "stop":   "s",
    "help":   "e",
    "good":   "g",
    "bad":    "b",
    "love":   "l",
    "i":      "i",
    "you":    "u",
    "ok":     "o",
    "water":  "w",
    "food":   "f",
    "i_eat_rice":     "1",
    "good_morning":   "2",
    "good_afternoon": "3",
    "how_are_you":    "4",
}

SAMPLES_PER_GESTURE = 200

# Landmark connections of the hand skeleton (for drawing)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm edge
]

# Pretrained hand-landmark model for the new "tasks" API
_TASK_MODEL_PATH = os.path.join("models", "hand_landmarker.task")
_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


# ----------------------------------------------------------------------
# MediaPipe hand detector (legacy + new API support)
# ----------------------------------------------------------------------

class HandDetector:
    """
    Detects one hand and returns its 21 landmarks as (x, y, z) tuples.

    Works with:
        * legacy API : mp.solutions.hands            (mediapipe <= 0.10.14)
        * new API    : mp.tasks.vision.HandLandmarker (mediapipe >= 0.10.15)
    """

    def __init__(self, max_hands: int = 1,
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5):
        self._legacy = hasattr(mp, "solutions")

        if self._legacy:
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self._ensure_task_model()
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=_TASK_MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(
                options)
            self._timestamp_ms = 0

    @staticmethod
    def _ensure_task_model():
        """Download hand_landmarker.task on first run (about 8 MB)."""
        if os.path.exists(_TASK_MODEL_PATH):
            return
        os.makedirs(os.path.dirname(_TASK_MODEL_PATH), exist_ok=True)
        print("[INFO] Downloading MediaPipe hand model (first run only)...")
        urllib.request.urlretrieve(_TASK_MODEL_URL, _TASK_MODEL_PATH)
        print(f"[INFO] Saved -> {_TASK_MODEL_PATH}")

    def find_hand(self, frame_bgr):
        """Return a list of 21 (x, y, z) tuples, or None if no hand found."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._legacy:
            results = self._hands.process(frame_rgb)
            if not results.multi_hand_landmarks:
                return None
            hand = results.multi_hand_landmarks[0]
            return [(lm.x, lm.y, lm.z) for lm in hand.landmark]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=frame_rgb)
        self._timestamp_ms += 33  # must increase monotonically (VIDEO mode)
        result = self._landmarker.detect_for_video(mp_image,
                                                   self._timestamp_ms)
        if not result.hand_landmarks:
            return None
        hand = result.hand_landmarks[0]
        return [(lm.x, lm.y, lm.z) for lm in hand]

    @staticmethod
    def draw(frame_bgr, landmarks):
        """Draw the hand skeleton on the frame (in place)."""
        if not landmarks:
            return
        h, w = frame_bgr.shape[:2]
        points = [(int(x * w), int(y * h)) for x, y, _ in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, points[a], points[b], (0, 200, 0), 2)
        for px, py in points:
            cv2.circle(frame_bgr, (px, py), 4, (0, 0, 255), -1)

    def close(self):
        if self._legacy:
            self._hands.close()
        else:
            self._landmarker.close()


# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------

def normalize_landmarks(landmarks):
    """
    Convert 21 (x, y, z) landmarks into a normalized 63-value feature vector.

    Steps:
        1. Translate: subtract the wrist (landmark 0) so position on screen
           does not matter.
        2. Scale: divide by the largest distance from the wrist so distance
           from the camera / hand size does not matter.
    """
    points = np.array(landmarks, dtype=np.float32)          # (21, 3)
    points = points - points[0]                              # translation

    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist                           # scale

    return points.flatten()                                  # (63,)


def extract_landmarks(detector: HandDetector, frame_bgr):
    """
    Detect a hand in the frame.

    Returns
    -------
    features  : normalized 63-value numpy vector, or None if no hand found
    landmarks : raw list of 21 (x, y, z) tuples for drawing, or None
    """
    landmarks = detector.find_hand(frame_bgr)
    if landmarks is None:
        return None, None
    return normalize_landmarks(landmarks), landmarks


# ----------------------------------------------------------------------
# Dataset helpers
# ----------------------------------------------------------------------

def ensure_dataset_dirs():
    """Create dataset/<gesture>/ folders if they do not exist."""
    for gesture in GESTURES:
        os.makedirs(os.path.join(DATASET_DIR, gesture), exist_ok=True)


def count_samples(gesture: str) -> int:
    """Number of samples already collected for a gesture."""
    folder = os.path.join(DATASET_DIR, gesture)
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".npy")])


def save_sample(gesture: str, features: np.ndarray) -> str:
    """Save one feature vector as dataset/<gesture>/NNN.npy and return path."""
    index = count_samples(gesture) + 1
    path = os.path.join(DATASET_DIR, gesture, f"{index:03d}.npy")
    np.save(path, features)
    return path


def load_dataset():
    """
    Load every .npy sample from the dataset folder.

    Returns
    -------
    X : (n_samples, 63) float32 array
    y : (n_samples,) array of string labels
    """
    X, y = [], []
    if not os.path.isdir(DATASET_DIR):
        return np.empty((0, NUM_FEATURES)), np.array([])

    for gesture in sorted(os.listdir(DATASET_DIR)):
        folder = os.path.join(DATASET_DIR, gesture)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".npy"):
                features = np.load(os.path.join(folder, fname))
                if features.shape == (NUM_FEATURES,):
                    X.append(features)
                    y.append(gesture)

    return np.array(X, dtype=np.float32), np.array(y)


def put_text(frame, text, org, scale=0.8, color=(0, 255, 0), thickness=2):
    """Readable text with a dark outline for OpenCV overlays."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)
