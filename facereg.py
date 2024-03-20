import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle

imgelon =face_recognition.load_image_file('elon.jpg')
imgelon = cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
#----------Finding face Location for drawing bounding boxes-------
face = face_recognition.face_locations(imgelon)[0]
copy = imgelon.copy()
#-------------------Drawing the Rectangle-------------------------
cv2.rectangle(copy, (face[3], face[0]),(face[1], face[2]), (255,0,255), 2)
#cv2.imshow('copy', copy)
#cv2.imshow('js',imgelon)
cv2.waitKey(0)

# lets test an image
#test = face_recognition.load_image_file('elon_2.jpg')
#test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
#test_encode = face_recognition.face_encodings(test)[0]
#print(face_recognition.compare_faces([train_encode],test_encode))
images = []
classNames = []
path="images"
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

    
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)


def login(name):
    print("login")
    print("welcome, ", name)

            
# take pictures from webcam 
cap  = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        print(matchIndex)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()
            y1,x2,y2,x1 = faceloc
            # since we scaled down by 4 times
            y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            login(name)
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
