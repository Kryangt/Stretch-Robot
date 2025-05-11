import cv2 as cv
import mediapipe as mp #a top-level module
import argparse
import math

args = argparse.ArgumentParser()
args.add_argument('--mode', default='webcam')
args.add_argument('--filepath', default="./None.jpg")

args = args.parse_args()

max_eye_distance_ratio = 0.05
#Gaze: whether people are looking at the camera
#Skeleton: Posture, orientation, depth camera

def process_image(img, face_detection):

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    # relative keypoints -- face landmarks
    # bounding box -- where is the face
    H, W, _ = img.shape
    if out.detections is not None:  # in case no face detected
        for detection in out.detections:
            location = detection.location_data
            bbox = location.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height


            #calculate the relative distance between two eyes
            #if the distance is big, then the face is facing the camera
            #if the distance is small, then the face is looking at the side
            face_score = faceCameraRate(location.relative_keypoints[1], location.relative_keypoints[0], w)


            # since x1, y1 are all relative points, convert to actual size
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # do some manipulation to the face
            img[y1: y1 + h, x1: x1 + w, :] = cv.blur(img[y1: y1 + h, x1: x1 + w, :],
                                                      (30, 30))  # this function generates a new image with blur
            img[y1: y1 + h, x1: x1 + w, :] = addInfo(face_score, img[y1: y1 + h, x1: x1 + w, :])
    return img


def addInfo(score, img):
    color = (255, 0, 0)
    thickness = 4
    H, W, _ = img.shape
    cv.rectangle(img, (0, 0), (W, H), color, thickness)

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_color = (255, 255, 255)
    thickness = 2
    position = (10, 20)

    cv.putText(img, f'{score:.2f}', position, font, font_scale, text_color, thickness, cv.LINE_AA)

    return img
#calculate the score of whether facing the camera dynamically
#set a reasonable ideal eye distance first
#if current eye distance is greater than the ideal one, modify the ideal eye distance
#Then, calculate the score
#If Looking at the side, score will be lower
#If Looking at camera, score will be higher
def faceCameraRate(left_eye_posit, right_eye_posit, width):
    x_distance = abs(right_eye_posit.x - left_eye_posit.x)
    y_distance = abs(right_eye_posit.y - left_eye_posit.y)

    relativeDistance = math.sqrt(pow(x_distance, 2) + pow(y_distance, 2))

    currentRatio = relativeDistance / width
    global max_eye_distance_ratio
    max_eye_distance_ratio = max(currentRatio, max_eye_distance_ratio)

    return currentRatio / max_eye_distance_ratio


mp_face_detector = mp.solutions.face_detection #mp.solutions is an attribute of mp, and also a module, face_detection is also a module
with mp_face_detector.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    #FaceDetection is an instance or an object or a model used to detect the faces in an image based on the parameters we passed in

    if args.mode in ['image']:
        #read image
        img = cv.imread(args.filepath)
        if(img is None):
            print("hello world")
        else:
            img = process_image(img, face_detection)

            cv.imshow('Face Detection', img)
            cv.waitKey(0)
    elif args.mode in ['webcam']:
        cap = cv.VideoCapture(0)

        ret, frame = cap.read() #capture each frame as an image
        while ret:
            frame = process_image(frame, face_detection)
            cv.imshow('Face Detection', frame)
            key = cv.waitKey(16)
            if key == 27 or key == ord('q'):
                break
            ret,frame = cap.read()

        cap.release() #release memory in the end

