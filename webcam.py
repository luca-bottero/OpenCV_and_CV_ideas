import cv2

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

def manipulate_frame(frame):
    '''
    This function acts on a cv frame
    '''

    pass


while True:
    ret, frame = cap.read()     #ret is aflag that indicates if the frame has been captured
    frame = cv2.resize(frame, None, fx=1., fy=1., interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    manip_res = manipulate_frame(frame)

    cv2.putText(frame, "OpenCV + Jurassic Park!!!", (10, 25), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    c = cv2.waitKey(1)
    if c == 27:
        break       #27 is ascii for ESC

cap.release()
cv2.destroyAllWindows()