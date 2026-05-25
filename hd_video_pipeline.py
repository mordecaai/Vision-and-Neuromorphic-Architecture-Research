import cv2
import numpy as np
from grid_window import GridWindow

# config
THRESHOLDS = [42, 127, 212]
FPS = 12
LOOP = False

# video processing
cap = cv2.VideoCapture("decoloredvideo.mp4")
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (48, 27), interpolation=cv2.INTER_AREA)
    quantized = np.digitize(small, THRESHOLDS).astype(np.uint8)
    frames.append(quantized)

cap.release()

spike_tensor = np.stack(frames)   # shape: (T, 27, 48), values 0-3
print(f"converted: {spike_tensor.shape[0]} frames")

# play in grid window
win = GridWindow(cols=48, rows=27, cell_px=14)
win.play(spike_tensor, fps=FPS, loop=LOOP)