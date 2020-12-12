import numpy as np
import cv2



#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#objects = cv.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]
#image            Matrix of the type CV_8U containing an image where objects
#                 are detected.  

#objects          Vector of rectangles where each rectangle contains the  
#                 detected object, the rectangles may be partially outside  
#                 the original image.  

#scaleFactor      Parameter specifying how much the image size is reduced 
#                 at each image scale.  

#minNeighbors     Parameter specifying how many neighbors each candidate  
#                 rectangle should have to retain it.

#flags            Parameter with the same meaning for an old cascade as in  
#                 the function cvHaarDetectObjects. It is not used for a  
#                 new cascade.

#minSize          Minimum possible object size. Objects smaller than that  
#                 are ignored.  

#maxSize          Maximum possible object size. Objects larger than that  
#                 are ignored. If maxSize == minSize model is evaluated  
#                 on single scale.



    for (x,y,w,h) in faces:
        #draw rectangle co-ord the coordiantes in red. Line width 2.
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        #crop of inside face
        roi_gray = gray[y:y+h, x:x+w]

        #colour crop of inside face
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #using img orifinal bgr colour foramt to display iamge
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
#allows escape from the loop.
