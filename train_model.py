"""
train_model.py  (Step 2 - Model Training)
-----------------------------------------
Loads the collected dataset, trains a machine learning classifier,
prints the evaluation report and saves the model to models/sign_model.pkl.

How to use:
    python train_model.py            # trains a Random Forest (default)
    python train_model.py --svm      # trains an SVM instead
"""

import os
import sys
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import load_dataset, MODEL_PATH


def main():
    use_svm = "--svm" in sys.argv

    print("=" * 50)
    print(" AI Sign Language Translator - Model Training")
    print("=" * 50)

    # ------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------
    print("[1/4] Loading dataset ...")
    X, y = load_dataset()

    if len(X) == 0:
        print("[ERROR] Dataset is empty. Run collect_data.py first.")
        return

    labels = sorted(set(y))
    print(f"       Dataset Loaded : {len(X)} samples, "
          f"{len(labels)} gestures -> {', '.join(l.upper() for l in labels)}")

    # ------------------------------------------------------------
    # 2. Train / test split (80 / 20)
    # ------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[2/4] Split          : {len(X_train)} train / {len(X_test)} test")

    # ------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------
    if use_svm:
        model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
        print("[3/4] Training Started (SVM) ...")
    else:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=None,
            random_state=42, n_jobs=-1,
        )
        print("[3/4] Training Started (Random Forest) ...")

    model.fit(X_train, y_train)

    # ------------------------------------------------------------
    # 4. Evaluate + save
    # ------------------------------------------------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"[4/4] Accuracy       : {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix (rows = actual, cols = predicted):")
    print(np.array2string(confusion_matrix(y_test, y_pred, labels=labels)))
    print("Labels order:", [l.upper() for l in labels])

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "labels": labels}, f)

    print(f"\nModel Saved -> {MODEL_PATH}")
    print("Next step:  python detect_sign.py")


if __name__ == "__main__":
    main()
