import cv2
import numpy as np
from Conv_AE_TF import AE_model_TF
import matplotlib.pyplot as plt


#Setup AE model

IMG_SHAPE = (28, 28, 1)
USE_LSTM  = True
BATCH_LEN = 4

ae_model = AE_model_TF(img_shape=IMG_SHAPE, lstm = USE_LSTM, batch_len = BATCH_LEN)

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

i = 0
frame_batch = []

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(frame, dsize=IMG_SHAPE[:-1], interpolation=cv2.INTER_LANCZOS4)

    frame_batch.append(resized)

    if i >= BATCH_LEN:
        frame_batch = frame_batch[1:]

        inp, out = ae_model.train_predict(np.array(frame_batch, dtype=int))
        #cv2.imshow(canny, np.hstack((inp[0,-1,:,:], out)))
        upscaled_out = cv2.resize(out, dsize=(320,240), interpolation=cv2.INTER_AREA)
        cv2.imshow(canny, upscaled_out)

        #print(inp.shape)

    i += 1
    #upscaled_out = cv2.resize(out, dsize=(640,480), interpolation=cv2.INTER_AREA)
    #upscaled_inp = cv2.resize(inp, dsize=(640,480), interpolation=cv2.INTER_AREA)
    #print(upscaled_inp)

       
    
    cv2.imshow('WebCam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()