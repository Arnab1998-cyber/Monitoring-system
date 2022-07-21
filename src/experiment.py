"""import winsound
frequency = 2500
duration = 1000
"""
import cv2
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib

detector = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("src\models\shape_predictor_68_face_landmarks.dat")


thresh = 0.20
flag = 0
frame_check = 20

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame1 = frame.copy()
    frame = imutils.resize(frame, width=450)
    print(frame.shape)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces  = detector(gray)
    for points in faces:
        shape = predict(gray, points)
        shape = face_utils.shape_to_np(shape) #converting to NumPy Array
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
            #print (flag)
            if flag >= frame_check:
                cv2.putText(frame1, "*********************ALERT!*********************", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            flag = 0

    cv2.imshow("Frame", frame1)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()
        cap.release()
        break
        
    
