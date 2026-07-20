"""
collect_data.py  (Step 1 - Dataset Collection)
----------------------------------------------
Opens the webcam and lets you record landmark samples for each gesture.

How to use:
    python collect_data.py

    1. Show a gesture with your hand in front of the camera.
    2. Press the key assigned to that gesture (see the on-screen list),
       e.g. press "H" while showing the HELLO sign.
    3. The script records 200 samples automatically while you hold the sign.
       Move your hand slightly (angle / distance) during recording so the
       model learns variations.
    4. Repeat for every gesture, then press Q to quit.

Samples are saved as dataset/<gesture>/001.npy, 002.npy, ...
Each sample is a normalized 63-value feature vector (21 landmarks x, y, z).
"""

import time

import cv2

from utils import (
    GESTURES, SAMPLES_PER_GESTURE,
    HandDetector, extract_landmarks,
    ensure_dataset_dirs, count_samples, save_sample, put_text,
)

KEY_TO_GESTURE = {key: gesture for gesture, key in GESTURES.items()}


def main():
    ensure_dataset_dirs()
    detector = HandDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open the webcam.")
        return

    print("=" * 50)
    print(" AI Sign Language Translator - Data Collection")
    print("=" * 50)
    print("Press the key of a gesture to start recording 200 samples.")
    for gesture, key in GESTURES.items():
        print(f"   [{key.upper()}] -> {gesture.upper():8s} "
              f"(collected: {count_samples(gesture)})")
    print("   [Q] -> Quit")
    print("=" * 50)

    recording_for = None      # gesture currently being recorded
    recorded_now = 0          # samples recorded in the current session

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror view feels natural
        features, hand_landmarks = extract_landmarks(detector, frame)
        detector.draw(frame, hand_landmarks)

        # ------------------------------------------------------------
        # Recording mode
        # ------------------------------------------------------------
        if recording_for is not None:
            total = count_samples(recording_for)

            if features is not None and total < SAMPLES_PER_GESTURE:
                save_sample(recording_for, features)
                recorded_now += 1
                total += 1

            put_text(frame, f"RECORDING: {recording_for.upper()}",
                     (10, 40), color=(0, 0, 255))
            put_text(frame, f"Samples: {total}/{SAMPLES_PER_GESTURE}",
                     (10, 80))

            if features is None:
                put_text(frame, "Show your hand!", (10, 120),
                         color=(0, 255, 255))

            if total >= SAMPLES_PER_GESTURE:
                print(f"[DONE] {recording_for.upper()} - "
                      f"{total} samples collected.")
                recording_for = None
                recorded_now = 0
                time.sleep(0.5)
        else:
            put_text(frame, "Press gesture key to record | Q = quit",
                     (10, 40), scale=0.7)
            if features is not None:
                put_text(frame, "Hand detected", (10, 80),
                         color=(255, 200, 0), scale=0.7)

        cv2.imshow("Data Collection - AI Sign Language Translator", frame)

        # ------------------------------------------------------------
        # Keyboard handling
        # ------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") and recording_for is None:
            break
        if key != 255 and recording_for is None:
            char = chr(key).lower()
            if char in KEY_TO_GESTURE:
                recording_for = KEY_TO_GESTURE[char]
                recorded_now = 0
                existing = count_samples(recording_for)
                if existing >= SAMPLES_PER_GESTURE:
                    print(f"[INFO] {recording_for.upper()} already has "
                          f"{existing} samples. Delete the folder to redo.")
                    recording_for = None
                else:
                    print(f"[REC ] Recording {recording_for.upper()} ... "
                          f"hold the sign in front of the camera.")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    print("\nCollection summary:")
    for gesture in GESTURES:
        print(f"   {gesture.upper():8s} : {count_samples(gesture)} samples")
    print("\nNext step:  python train_model.py")


if __name__ == "__main__":
    main()
