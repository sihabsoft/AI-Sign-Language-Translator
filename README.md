# AI Powered Sign Language Translator

A computer-based Python application that recognizes American Sign Language (ASL)
hand gestures through a standard webcam and translates them into text in real
time — building complete sentences from individual signs.

**Type:** Desktop application (Terminal + Webcam) · **No framework, raw Python**

```
Camera Started...
Detected Gesture : HELLO      Confidence : 99%
Sentence : HELLO
----------------------------------------
Detected Gesture : HOW        Confidence : 97%
Sentence : HELLO HOW
----------------------------------------
Detected Gesture : YOU        Confidence : 98%
Sentence : HELLO HOW YOU
```

## How It Works

```
User → Webcam → OpenCV (video capture) → MediaPipe (hand detection)
     → 21 Hand Landmarks → 63 Feature Vector → ML Model (Random Forest / SVM)
     → Gesture Prediction → Terminal Output → Sentence Builder
```

MediaPipe finds 21 landmark points on the hand and gives the (x, y, z)
coordinate of each point → 21 × 3 = **63 numeric features**. The model never
sees the raw image; it classifies gestures purely from these 63 numbers, which
makes it fast and robust to lighting/background changes. Features are
normalized relative to the wrist so hand position and camera distance do not
affect the prediction.

## Project Structure

```
AI-Sign-Language-Translator/
├── dataset/              # created automatically; one folder per gesture
│     ├── hello/001.npy ...
│     └── ...
├── models/
│     └── sign_model.pkl  # trained classifier (created by train_model.py)
├── collect_data.py       # Step 1: dataset collection
├── train_model.py        # Step 2: model training
├── detect_sign.py        # Step 3: real-time detection + sentence builder
├── utils.py              # shared helpers (MediaPipe wrapper, features, dataset)
├── requirements.txt
└── README.md
```

## Setup

Requires **Python 3.9 – 3.12** and a webcam.

```bash
pip install -r requirements.txt
```

(`pyttsx3` is optional — only needed for voice output.)

## Usage

### Step 1 — Collect the dataset

```bash
python collect_data.py
```

The webcam opens. Show a gesture and press its key (shown on screen and in the
terminal, e.g. **H** for HELLO). The script records **200 samples** while you
hold the sign — move your hand slightly (angle/distance) during recording so
the model learns variations. Repeat for all gestures, then press **Q**.

| Key | Gesture | Key | Gesture | Key | Gesture |
|-----|---------|-----|---------|-----|---------|
| H | Hello | S | Stop | I | I |
| T | Thanks | E | Help | U | You |
| Y | Yes | G | Good | O | OK |
| N | No | B | Bad | W | Water |
| P | Please | L | Love | F | Food |
| A | Eat | R | Rice | M | Morning |
| D | Afternoon | Q | How | V | Are |

Combine the words above to build full sentences, since each gesture maps to a
single word and the sentence builder joins them in the order you sign:

- **I eat rice** → sign `I` + `Eat` + `Rice`
- **Good morning** → sign `Good` + `Morning`
- **Good afternoon** → sign `Good` + `Afternoon`
- **How are you** → sign `How` + `Are` + `You`

### Step 2 — Train the model

```bash
python train_model.py          # Random Forest (default)
python train_model.py --svm    # SVM alternative
```

Output: accuracy (expect ~95–99%), classification report, confusion matrix,
and the saved model at `models/sign_model.pkl`.

### Step 3 — Real-time translation

```bash
python detect_sign.py            # normal mode
python detect_sign.py --voice    # also speaks each word (pyttsx3)
```

| Key | Action |
|-----|--------|
| R | Reset the current sentence |
| S | Save the sentence to `conversation.txt` |
| Q | Quit |

A gesture is added to the sentence only when it stays **stable for ~15 frames
with ≥80% confidence**, and a short cooldown prevents the same word from
repeating accidentally.

## Tips for High Accuracy

- Collect data in the **same lighting** you will demo in, with a plain background.
- During collection, vary the hand angle and distance slightly.
- Keep gestures visually distinct; if two signs confuse the model, check the
  confusion matrix printed by `train_model.py`.
- To redo one gesture, delete its folder inside `dataset/` and collect again.

## Viva Quick Notes

- **OpenCV** — open-source computer vision library; here it captures webcam
  frames and draws the UI overlay.
- **MediaPipe** — Google's ML framework; its Hands solution detects the hand
  and returns 21 3D landmarks per hand in real time.
- **Landmark** — a key point on the hand (wrist, knuckles, fingertips).
- **Why ML?** — hand-written if/else rules cannot cover natural variation in
  hand shapes; a classifier learns the pattern from examples.
- **Random Forest** — an ensemble of decision trees voting together; accurate,
  fast, and resistant to overfitting on small datasets.
- **Accuracy measurement** — 80/20 train/test split; accuracy, classification
  report and confusion matrix on the unseen 20%.
- **Overfitting** — model memorizes training data but fails on new data;
  detected when train accuracy ≫ test accuracy; reduced here by feature
  normalization and dataset variation.
- **Real-time prediction** — every frame: landmarks → 63 features →
  `model.predict_proba` → stability check → sentence builder.

## Future Scope

Dynamic gestures (LSTM), Bangla Sign Language support, emotion detection,
GUI (Tkinter/PyQt), two-hand gestures, mobile version, CNN on raw images.
