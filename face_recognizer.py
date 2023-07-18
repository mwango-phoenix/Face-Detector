import cv2
import urllib.request
import numpy as np

#classifiers
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #for detecting faces in the image
eyes = cv2.CascadeClassifier('haarcascade_eye.xml') 

# get the image link from the user
image_url = input("Please enter the image URL: ")

try:
    # download the image from the URL
    resp = urllib.request.urlopen(image_url)
    # reads image data and converts into NumPy array of unsigned 8-bit ints
    img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    #convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # run face classifier with scale factor 1.1 and minimum number of neighbour rectangles 4
    classify = face.detectMultiScale(gray_img, 1.1, 4)

    # create rectangles of model from bounding box
    for (x,y, w, h) in classify:
        #draws rectangle around detected face
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 0), 2)
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_colour = img[y:y+h, x:x+w]
        # runs eye detection classifier on grayscale region of interest
        eyes_create = eyes.detectMultiScale(roi_gray)
        for (xe, ye, we, he) in eyes_create:
            #create bounding box on eyes
            cv2.rectangle(roi_colour, (xe, ye), (xe+we, ye+he), (0, 127, 255), 2)

    #display image
    cv2.imshow('img', img)
    cv2.waitKey()

except Exception as e:
    print("Error occurred: ", e)
