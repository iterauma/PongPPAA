# Gesture Games — CS Open Day Stand

Two gesture-controlled games built with [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and [Pygame](https://www.pygame.org/).  
Players use their **bare hands** in front of a webcam — no controllers needed.

| Game | File | Players |
|---|---|---|
| Pong | `pong.py` | 2 (one hand each) |
| Whack-a-Mole | `whackamole.py` | 1 or 2 (one hand each) |

Both games include a **live debug panel** showing the raw camera feed with hand skeleton overlays, so bystanders can see exactly what the computer is detecting.

---

## Requirements

- Python **3.9 or newer**
- A USB or built-in webcam
- Reasonably even lighting (outdoors is fine; avoid pointing the camera directly into the sun)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/iterauma/PongPPAA.git
cd PongPPAA
```

### 2. Create and activate a virtual environment

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the hand landmark model

The model file (`hand_landmarker.task`, ~26 MB) is downloaded automatically on first run and saved in the project folder. You need an internet connection the first time only.

---

## Running the games

### Pong

```bash
python pong.py
```

- Stand ~1–1.5 m from the camera
- **Left player** — raise your **left hand** to control the left paddle
- **Right player** — raise your **right hand** to control the right paddle
- First to **5 points** wins
- Press **SPACE** on the win screen to play again, **ESC** to quit

### Whack-a-Mole

```bash
python whackamole.py
```

- One or two players can play simultaneously
- Move your palm **over a mole** to whack it — the hit registers the moment your hand enters the mole's radius
- **+10 pts** for a successful whack, **−5 pts** for hitting an empty hole
- Moles speed up every 5 whacks
- You have **60 seconds** — highest score wins
- Press **SPACE** on the end screen to play again, **ESC** to quit

---

## How it works

Hand tracking is powered by **MediaPipe Hand Landmarker**, which detects 21 keypoints per hand in real time. Both games use **landmark 9** (the base of the middle finger, roughly the palm centre) as the control point.

The debug panel on the right side of the window shows:
- The live camera feed annotated with the full hand skeleton
- The control point (landmark 9) highlighted in red
- Live normalised coordinates per detected hand

---

## Troubleshooting

**Blank or broken debug window / Qt font errors**  
This is a known issue with some Linux builds of `opencv-python`. Both games render the debug feed inside the Pygame window directly, so no separate OpenCV window is used and this error is harmless — the game will work normally.

**MediaPipe log spam in the terminal**  
Lines like `inference_feedback_manager` or `landmark_projection_calculator` are internal TFLite warnings and can be safely ignored.

**Wrong camera selected**  
If the game opens the wrong camera, change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` (or `2`, etc.) near the top of the relevant file.

**Laggy / jittery detection outdoors**  
- Ensure the sun is behind or to the side of the players, not shining into the lens
- A 1080p 60 fps USB webcam (e.g. Logitech C920/C922) performs significantly better than a built-in laptop camera in variable outdoor light

---

## Project structure

```
gesture-games/
├── pong.py
├── whackamole.py
├── requirements.txt
└── README.md
```

`hand_landmarker.task` will appear here after the first run.
