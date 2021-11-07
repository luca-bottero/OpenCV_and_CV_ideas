import cv2
import numpy as np
from Conv_AE import AE_model
import matplotlib.pyplot as plt


#Setup AE model

IMG_SHAPE = (100, 100, 3)

ae_model = AE_model(img_shape=IMG_SHAPE)


# Setup camera
cap = cv2.VideoCapture(0)
# Set a smaller resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def nothing(x):
    pass


canny = "Canny"
cv2.namedWindow(canny)
cv2.createTrackbar('Threshold 1', canny, 0, 255, nothing)
cv2.createTrackbar('Threshold 2', canny, 0, 255, nothing)


while True:
    # Capture frame-by-frame
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    #t1 = cv2.getTrackbarPos('Threshold 1', canny)
    #t2 = cv2.getTrackbarPos('Threshold 2', canny)
    #gb = cv2.GaussianBlur(frame, (5, 5), 0)
    #can = cv2.Canny(gb, t1, t2)

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(frame, dsize=IMG_SHAPE[:-1], interpolation=cv2.INTER_LANCZOS4)
    #print(resized)
    
    inp, out = ae_model.train_predict(np.array(resized, dtype=int))

    #upscaled_out = cv2.resize(out, dsize=(640,480), interpolation=cv2.INTER_AREA)
    #upscaled_inp = cv2.resize(inp, dsize=(640,480), interpolation=cv2.INTER_AREA)
    #print(upscaled_inp)

    cv2.imshow(canny, np.hstack((inp[0], out)))
    #cv2.imshow(canny, np.hstack((resized, inp[0], out)))
    #cv2.imshow(canny, resized)
    #cv2.imshow(canny, (inp[0]/255))
    #print(inp[0])
    




    #frame[np.where(can)] = 255
    cv2.imshow('WebCam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()