import cv2 as cv
import mediapipe as mp #a top-level module

img_path = "./TestImg.jpg"
img = cv.imread(img_path)

#detect faces
mp_face_detector = mp.solutions.face_detection #mp.solutions is an attribute of mp, and also a module, face_detection is also a module
with mp_face_detector.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    #FaceDetection is an instance or an object or a model used to detect the faces in an image based on the parameters we passed in
    img_rgb = cv.cvtcolor(img, cv.COLOR_BGR2RGB)