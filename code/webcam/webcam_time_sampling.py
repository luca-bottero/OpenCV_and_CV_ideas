import cv2
import os
import time

out_path = '../../results/time_sampling/'
time_lag = 1    #seconds


    

subj_name = input('Insert subject name\n')

cap = cv2.VideoCapture(2)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")


prev = time.time()

while True:
    ret, frame = cap.read()     #ret is aflag that indicates if the frame has been captured
    frame = cv2.resize(frame, None, fx=1., fy=1., interpolation=cv2.INTER_AREA)
    cv2.imshow('Frame', frame)

    if ret:
        if (s:= round(time.time() - prev)) >= time_lag:
            prev = s
            filename = out_path + subj_name + str(round(s)) + '.png'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            cv2.imwrite(filename, frame)

    cv2.putText(frame, "OpenCV + Jurassic Park!!!", (10, 25), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    c = cv2.waitKey(1)
    if c == 27:
        break       #27 is ascii for ESC

cap.release()
cv2.destroyAllWindows()