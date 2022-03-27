



# Import the necessary libraries
import numpy as np
import cv2 
import matplotlib.pyplot as plt

video_device = "/dev/video0"
cap = cv2.VideoCapture(video_device)
w = 500
h = w
cx = w/2
cy = h/2


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

haar_cascade_face = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

def get_faces():
    ret, frame = cap.read()
    inp = cv2.resize(frame, (w , h))

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rects = haar_cascade_face.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors = 5)
    
    for (ymin,xmin,ymax,xmax) in faces_rects:
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        img = cv2.circle(frame, (int(x),int(y)), radius=1, color=(255, 0, 0), thickness=-1)
        plt.imshow(convertToRGB(img))

while True:
    get_faces()
    if cv2.waitKey(0) & 0xFF == ord('q'):
            break




