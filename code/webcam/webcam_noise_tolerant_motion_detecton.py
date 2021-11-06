# Source: https://www.learnpythonwithrune.org/opencv-and-python-simple-noise-tolerant-motion-detector/

import cv2
import numpy as np
import imutils
import time
from collections import deque


# Input to Step 5: Helper function
# Calculate the foreground frame based on frames
def get_movement(frames, shape):
    movement_frame = np.zeros(shape, dtype='float32')
    i = 0
    for f in frames:
        i += 1
        movement_frame += f * i
    movement_frame = movement_frame / ((1 + i) / 2 * i)
    movement_frame[movement_frame > 254] = 255
    return movement_frame


# Input to Step 5: Helper function
# Calculate the background frame based on frames
# This function has obvious improvement potential
# - Could avoid to recalculate full list every time
def get_background(frames, shape):
    bg = np.zeros(shape, dtype='float32')
    for frame in frames:
        bg += frame
    bg /= len(frames)
    bg[bg > 254] = 255
    return bg


# Detect and return boxes of moving parts
def detect(frame, bg_frames, fg_frames, threshold=40, min_box=200):
    # Step 3-4: Add the frame to the our list of foreground and background frames
    fg_frames.append(frame)
    bg_frames.append(frame)

    # Input to Step 5: Calculate the foreground and background frame based on the lists
    fg_frame = get_movement(list(fg_frames), frame.shape)
    bg_frame = get_background(list(bg_frames), frame.shape)

    # Step 5: Calculate the difference to detect movement
    movement = cv2.absdiff(fg_frame, bg_frame)
    movement[movement < threshold] = 0
    movement[movement > 0] = 254
    movement = movement.astype('uint8')
    movement = cv2.cvtColor(movement, cv2.COLOR_BGR2GRAY)
    movement[movement > 0] = 254
    # As we don't return the movement frame, we show it here for debug purposes
    # Should be removed before release
    cv2.imshow('Movement', movement)

    # Step 6: Find the list of contours
    contours = cv2.findContours(movement, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Step 7: Convert them to boxes
    boxes = []
    for contour in contours:
        # Ignore small boxes
        if cv2.contourArea(contour) < min_box:
            continue
        # Convert the contour to a box and append it to the list
        box = cv2.boundingRect(contour)
        boxes.append(box)

    return boxes


def main(width=640, height=480, scale_factor=2):
    # Create the buffer of our lists
    bg_frames = deque(maxlen=30)
    fg_frames = deque(maxlen=10)

    # Get the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # We want to see how many frames per second we process
    last_time = time.time()
    while True:
        # Step 0: Read the webcam frame (ignore return code)
        _, frame = cap.read()

        # Resize the frame
        frame = cv2.resize(frame, (width, height))
        # Step 1: Scale down to improve speed (only takes integer scale factors)
        work_frame = cv2.resize(frame, (width // scale_factor, height // scale_factor))
        # Step 2: Blur it and convert the frame to float32
        work_frame = cv2.GaussianBlur(work_frame, (5, 5), 0)
        work_frame_f32 = work_frame.astype('float32')

        # Step 3-7 (steps in function): Detect all the boxes around the moving parts
        boxes = detect(work_frame_f32, bg_frames, fg_frames)

        # Step 8: Draw all boxes (remember to scale back up)
        for x, y, w, h in boxes:
            cv2.rectangle(frame, (x * scale_factor, y * scale_factor), ((x + w) * scale_factor, (y + h) * scale_factor),
                          (0, 255, 0), 2)

        # Add the Frames Per Second (FPS) and show frame
        text = "FPS:" + str(int(1 / (time.time() - last_time)))
        last_time = time.time()
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()