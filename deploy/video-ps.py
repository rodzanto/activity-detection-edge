from datetime import datetime
import numpy as np
import cv2
import threading
import os

# The videos will be of VIDEO_DURATION seconds
VIDEO_DURATION = 5
VIDEO_FILENAME = './tmp/video.avi'

cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')

def daemon():
    threading.Timer(VIDEO_DURATION, daemon).start()
    start = datetime.now()
    dt_string = start.strftime("%Y%m%d_%H%M%S%f")[:-3]
    os.rename(VIDEO_FILENAME, VIDEO_FILENAME + dt_string + '.avi')
    out = cv2.VideoWriter(VIDEO_FILENAME, fourcc, 20.0, size)
    while (cap.isOpened() and (datetime.now() - start).total_seconds() < VIDEO_DURATION):
        ret, frame = cap.read()
        if ret==True:
            out.write(frame)
        else:
            break
    out.release()

daemon()
