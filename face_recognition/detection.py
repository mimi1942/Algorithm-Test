import numpy as np
import cv2
# Create the haar cascade
cascPath = '/home/snu/opencv/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_alt.xml'
# cascPath = 'detect/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
cap = cv2.VideoCapture(0)

def detect_faces_show(fpath):
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    print("Found %d faces!" % len(faces))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)

while(True):
    ret, img = cap.read()
    if ret is True :
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else :
        continue
    detect_faces_show(img)

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
