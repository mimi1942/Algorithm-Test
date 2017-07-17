import numpy as np
import cv2
# Create the haar cascade
cascPath = '/home/snu/opencv/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_alt.xml'
# cascPath = 'detect/haarcascade_fqrontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
cap = cv2.VideoCapture(0)

def save_faces_show(fpath):
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

        #incrementing sample number
        global sampleNum
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) +".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('frame',img)


Id = raw_input('enter your id')
sampleNum=0
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''
    if ret is True :
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else :
        continue
    '''

    save_faces_show(img)

    cv2.imshow('frame',img)
    #wait for 100 miliseconds
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>20:
        break

cap.release()
cv2.destroyAllWindows()
