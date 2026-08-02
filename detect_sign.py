"""
detect_sign.py  (Step 3 - Real-Time Detection & Sentence Builder)
-----------------------------------------------------------------
Loads the trained model, opens the webcam and translates hand gestures
into text in real time. Confirmed gestures are appended to a sentence.

How to use:
    python detect_sign.py            # normal mode
    python detect_sign.py --voice    # also speak the sentence (needs pyttsx3)

Keyboard controls (while the camera window is focused):
    R : Reset the current sentence
    S : Save the current sentence to conversation.txt
    Q : Quit
"""

import os
import sys
import time
import pickle
from collections import deque, Counter

import cv2
import numpy as np

from utils import HandDetector, extract_landmarks, put_text, MODEL_PATH

# ----------------------------------------------------------------------
# Tuning parameters
# ----------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.80   # minimum probability to accept a prediction
STABLE_FRAMES = 15            # gesture must dominate the last N frames
COOLDOWN_SECONDS = 1.5        # wait before the same word can repeat
SAVE_FILE = "conversation.txt"


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model not found. Run train_model.py first.")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["labels"]


def init_voice(enabled: bool):
    """Return a pyttsx3 engine or None."""
    if not enabled:
        return None
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        return engine
    except Exception as exc:  # pragma: no cover - optional feature
        print(f"[WARN] Voice output unavailable ({exc}). Continuing silently.")
        return None


def speak(engine, text: str):
    if engine is not None and text:
        engine.say(text)
        engine.runAndWait()


def main():
    voice_engine = init_voice("--voice" in sys.argv)
    model, _labels = load_model()
    detector = HandDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open the webcam.")
        return

    print("=" * 50)
    print(" AI Sign Language Translator - Live Detection")
    print("=" * 50)
    print("Camera Started...")
    print("Controls: R = reset sentence | S = save | Q = quit\n")

    recent = deque(maxlen=STABLE_FRAMES)   # recent predicted labels
    sentence = []                          # confirmed words
    last_word = None
    last_word_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        features, hand_landmarks = extract_landmarks(detector, frame)
        detector.draw(frame, hand_landmarks)

        current_label, current_conf = None, 0.0

        # ------------------------------------------------------------
        # Prediction
        # ------------------------------------------------------------
        if features is not None:
            probs = model.predict_proba([features])[0]
            best = int(np.argmax(probs))
            current_conf = float(probs[best])
            current_label = model.classes_[best]

            if current_conf >= CONFIDENCE_THRESHOLD:
                recent.append(current_label)
            else:
                recent.append(None)
        else:
            recent.append(None)

        # ------------------------------------------------------------
        # Sentence builder: accept a word only when it is stable
        # ------------------------------------------------------------
        if len(recent) == STABLE_FRAMES:
            counts = Counter(w for w in recent if w is not None)
            if counts:
                word, freq = counts.most_common(1)[0]
                stable = freq >= int(STABLE_FRAMES * 0.8)
                now = time.time()
                not_repeat = (word != last_word
                              or now - last_word_time > COOLDOWN_SECONDS)

                if stable and not_repeat:
                    sentence.append(word.upper())
                    last_word, last_word_time = word, now
                    recent.clear()

                    print(f"Detected Gesture : {word.upper():10s} "
                          f"Confidence : {current_conf * 100:.0f}%")
                    print(f"Sentence : {' '.join(sentence)}")
                    print("-" * 40)
                    speak(voice_engine, word)

        # ------------------------------------------------------------
        # Overlay
        # ------------------------------------------------------------
        if current_label is not None:
            color = (0, 255, 0) if current_conf >= CONFIDENCE_THRESHOLD \
                else (0, 255, 255)
            put_text(frame, f"{current_label.upper()} "
                            f"({current_conf * 100:.0f}%)", (10, 45),
                     scale=1.0, color=color)
        else:
            put_text(frame, "No hand detected", (10, 45),
                     scale=0.8, color=(0, 0, 255))

        put_text(frame, "Sentence: " + " ".join(sentence[-6:]),
                 (10, frame.shape[0] - 20), scale=0.7,
                 color=(255, 255, 255))
        put_text(frame, "R=Reset  S=Save  Q=Quit", (10, 85),
                 scale=0.6, color=(200, 200, 200), thickness=1)

        cv2.imshow("AI Sign Language Translator", frame)

        # ------------------------------------------------------------
        # Keyboard controls
        # ------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            sentence.clear()
            last_word = None
            print("Sentence Cleared")
            print("-" * 40)
        elif key == ord("s"):
            text = " ".join(sentence)
            with open(SAVE_FILE, "a", encoding="utf-8") as f:
                f.write(text + "\n")
            print(f"Sentence Saved -> {SAVE_FILE} : {text}")
            print("-" * 40)
            speak(voice_engine, text)

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    final_text = " ".join(sentence)
    if final_text:
        print(f"\nFinal Sentence : {final_text}")
    print("Goodbye!")


if __name__ == "__main__":
    main()
