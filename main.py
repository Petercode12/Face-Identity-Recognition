import cv2
import numpy as np
import face_recognition
import matplotlib.pyplot as plt

#Test
image = face_recognition.load_image_file('ImagesBasic/Tom and Stark.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
face_loc = face_recognition.face_locations(image)
encode_Face = face_recognition.face_encodings(image)
print(face_loc)
print(encode_Face)
for face in face_loc:
    y1, x2, y2, x1 = face
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('Tom and Stark', image)
cv2.waitKey(0)
'''
#Step 1: Loading images & converting it into RGB
imgElon = face_recognition.load_image_file('ImagesBasic/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('ImagesBasic/Elon Test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

#Step 2: Finding faces in our images and then finding their encoding
faceLoc = face_recognition.face_locations(imgElon)[0] #Location = (Top, Right, Bottom, Left)
encodeElon = face_recognition.face_encodings(imgElon)[0] #128 measurements
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,255), 2) #Draw rectangle on face

faceLocTest = face_recognition.face_locations(imgTest)[0] #Location = (Top, Right, Bottom, Left)
encodeTest = face_recognition.face_encodings(imgTest)[0] #128 measurements
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,0,255), 2) #Draw rectangle on face

#Step 3: Use encodings to compare faces (actually compare these encodings) and find the distance between them with SVM
results = face_recognition.compare_faces([encodeElon], encodeTest) #Return a list of True/False (Ex: [True, False])
faceDis = face_recognition.face_distance([encodeElon], encodeTest) #To know how much they match, we need to calculate distance
print(results, faceDis)

#Final step: Label the faces
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)


cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)
'''
