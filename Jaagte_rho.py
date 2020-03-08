import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
alarm = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

model = load_model('cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
rate = 0
hicc = 2
rpred = [99]
lpred = [99]

while (True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    le = leye.detectMultiScale(gray)
    re = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in re:
        right_eye = frame[y:y + h, x:x + w]
        count = count + 1
        right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        right_eye = cv2.resize(right_eye, (24, 24))
        right_eye = right_eye / 255
        right_eye = right_eye.reshape(24, 24, -1)
        right_eye = np.expand_dims(right_eye, axis=0)
        rpred = model.predict_classes(right_eye)
        if (rpred[0] == 1):
            lbl = 'Open'
        if (rpred[0] == 0):
            lbl = 'Closed'
        break

    for (x, y, w, h) in le:
        left_eye = frame[y:y + h, x:x + w]
        count = count + 1
        left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        left_eye = cv2.resize(left_eye, (24, 24))
        left_eye = left_eye / 255
        left_eye = left_eye.reshape(24, 24, -1)
        left_eye = np.expand_dims(left_eye, axis=0)
        lpred = model.predict_classes(left_eye)
        if (lpred[0] == 1):
            lbl = 'Open'
        if (lpred[0] == 0):
            lbl = 'Closed'
        break

    if (rpred[0] == 0 and lpred[0] == 0):
        rate = rate + 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        rate = rate - 5
        #
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if (rate < 0):
        rate = 0
    cv2.putText(frame, 'rate:' + str(rate), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if (rate > 15):
        # person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            alarm.play()

        except:  # isplaying = False
            pass
        if (hicc < 16):
            hicc = hicc + 2
        else:
            hicc = hicc - 2
            if (hicc < 2):
                hicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), hicc)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
