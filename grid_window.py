"""
run_animation.py — process a video → spike tensor → play in grid window
Edit the CONFIG block at the top, then: python run_animation.py
"""

import cv2
import numpy as np
from grid_window import GridWindow

# ── CONFIG ────────────────────────────────────────────────────────────────────
VIDEO_PATH   = "decoloredvideo.mp4"   # path to your video file
GRID_COLS    = 48                      # must match resize width below
GRID_ROWS    = 27                      # must match resize height below
CELL_PX      = 14                      # pixel size of each grid cell
FPS          = 12                      # playback speed
LOOP         = False                   # set True to loop forever
THRESHOLDS   = [42, 127, 212]          # grayscale quantization thresholds
MAX_FRAMES   = None                    # set e.g. 100 to cap frame count
# ─────────────────────────────────────────────────────────────────────────────


def load_tensor(path, cols, rows, thresholds, max_frames=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small     = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
        quantized = np.digitize(small, thresholds).astype(np.uint8)
        frames.append(quantized)
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    if not frames:
        raise ValueError("No frames extracted — check video path.")

    tensor = np.stack(frames)                # (T, H, W)
    print(f"Loaded {tensor.shape[0]} frames  ({rows}x{cols} grid)")
    return tensor


if __name__ == "__main__":
    tensor = load_tensor(VIDEO_PATH, GRID_COLS, GRID_ROWS, THRESHOLDS, MAX_FRAMES)

    win = GridWindow(cols=GRID_COLS, rows=GRID_ROWS, cell_px=CELL_PX)
    win.play(tensor, fps=FPS, loop=LOOP)