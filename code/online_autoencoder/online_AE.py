import cv2
import numpy as np

# Setup camera
cap = cv2.VideoCapture(2)
# Set a smaller resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def nothing(x):
    pass


rec_window = "rec_window"
cv2.namedWindow(rec_window)
cv2.createTrackbar('Trackbar 1', rec_window, 0, 255, nothing)
cv2.createTrackbar('Threshold 2', rec_window, 0, 255, nothing)

while True:
    # Capture frame-by-frame
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    t1 = cv2.getTrackbarPos('Trackbar 1', rec_window)
    t2 = cv2.getTrackbarPos('Threshold 2', rec_window)
    gb = cv2.GaussianBlur(frame, (5, 5), 0)
    can = cv2.rec_window(gb, t1, t2)

    cv2.imshow(rec_window, can)

    frame[np.where(can)] = 255
    cv2.imshow('WebCam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()