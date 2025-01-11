from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from random import randrange
from moviepy.video.io.bindings import mplfig_to_npimage
import time


# Parameters

frame_check = 45
thresh = 0.25
window_l = 852
window_w = 480
graph_speed = .01



heart_rate = 0


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


index = count()
x_vals = []
y_red = []
y_blue = []

img = plt.imread('white.jpg')
img_ = np.copy(img)

ui_bg = np.copy(img)
ui_bg = cv2.resize(ui_bg, (1920, 1080))

data_monitor = np.copy(img)
data_monitor = cv2.resize(data_monitor, (1800, 450))
data_monitor_ = np.copy(data_monitor)

# cap=cv2.VideoCapture(0)                  #For webcam
cap = cv2.VideoCapture("/content/video.mp4")  # For recorded video 

flag = 0
sleeped = False

while True:
    # time.sleep(graph_speed)
    img = img_
    data_monitor = np.copy(data_monitor_)
    red_lines = []
    blue_lines = []

    red = randrange(200, 300)
    blue = randrange(200, 300)

    if(len(x_vals) < window_l/5):
        temp = next(index)
        x_vals.append(temp * 5)
        y_red.append(red)
        y_blue.append(blue)

    else:

        y_red.pop(0)
        y_blue.pop(0)
        y_red.append(red)
        y_blue.append(blue)

    

    img = cv2.resize(img, (window_l, window_w))

    red_lines.append(x_vals)
    red_lines.append(y_red)

    blue_lines.append(x_vals)
    blue_lines.append(y_blue)

    red_lines = np.array(red_lines).T
    blue_lines = np.array(blue_lines).T

    # lines = np.array([[10, 20], [20, 40], [30, 60], [40, 80], [50, 100]])
    red_lines.reshape((-1, 1, 2))
    blue_lines.reshape((-1, 1, 2))

    cv2.polylines(img, [red_lines], False, (255, 0, 0), 2)
    cv2.polylines(img, [blue_lines], False, (0, 0, 255), 2)

    cv2.putText(img, "Radar Data Live Plot", (300, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)



    ret, frame = cap.read()
    
    frame = imutils.resize(frame, width=window_w)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if ear < thresh:
            flag += 1
            # print(flag)
            if flag >= frame_check:
                sleeped = True
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, frame.shape[0]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            flag = 0
            sleeped = False

    frame = cv2.resize(frame, (window_l, window_w))
  

    s_img = cv2.imread("smaller_image.png")
    l_img = cv2.imread("larger_image.jpg")

    # data_monitor = data_monitor_

    if(sleeped):
        cv2.putText(data_monitor, "Is the driver sleeping? : Yes", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        cv2.putText(data_monitor, "Is the driver sleeping? : No", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    heart_rate = 0
    for i in y_red:
        if(i > 260):
            heart_rate+=1



    cv2.putText(data_monitor, "Heart Rate : {}".format(heart_rate), (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    # cv2.putText(data_monitor, "Heart Rate : {}".format(heart_rate), (10, 20),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)







    x_offset = 10
    y_offset = 10
    ui_bg[y_offset:y_offset+frame.shape[0], x_offset:x_offset+frame.shape[1]] = frame
    ui_bg[y_offset:y_offset+img.shape[0], x_offset + window_l + 100:x_offset + window_l + 100 +img.shape[1]] = img
    ui_bg[y_offset + window_w + 100:y_offset + window_w + 100 +data_monitor.shape[0], x_offset:x_offset+data_monitor.shape[1]] = data_monitor
    
    cv2.imshow("Frame", ui_bg)

# 	out.write(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()
        cap.release()
        break
