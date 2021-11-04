# Source: https://www.learnpythonwithrune.org/opencv-python-webcam-create-a-ghost-effect/

import cv2

# A global variable with the ghost effect
ghost_effect = 0.0
# Setup a window that can be referenced
window = "Webcam"
cv2.namedWindow(window)


# Used by the trackbar to change the ghost effect
def on_ghost_trackbar(val):
    global ghost_effect
    global window

    ghost_effect = val / 100.0
    cv2.setTrackbarPos("Shadow", window, val)


# Capture the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create a trackbar
cv2.createTrackbar("Ghost effect", window, 0, 100, on_ghost_trackbar)


# Get the first frame
_, last_frame = cap.read()
while True:
    # Get the next frame
    _, frame = cap.read()

    # Add the ghost effect
    if frame.shape == last_frame.shape:
        frame = cv2.addWeighted(src1=frame, alpha=1 - ghost_effect, src2=last_frame, beta=ghost_effect, gamma=0.0)

    # Update the frame in the window
    cv2.imshow(window, frame)
    
    # Update last_frame
    last_frame = frame

    # Check if q is pressed, terminate if so
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy windows
cap.release()
cv2.destroyAllWindows()