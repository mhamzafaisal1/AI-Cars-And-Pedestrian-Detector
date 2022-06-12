# TO RUN ON WINDOWS RUN THE COMMAND py <filename>


import cv2
from cv2 import COLOR_BGR2GRAY
from random import randrange

img_file = 'CarImage.png'

classifier_file = 'car_detector.xml'

# create open-cv image
img = cv2.imread(img_file)

# create classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# convert to grayscale (needed for haar cascade algorithm)
grayscale_img = cv2.cvtColor(img, COLOR_BGR2GRAY)

# detect cars
cars = car_tracker.detectMultiScale(grayscale_img)

# draw the rectangle around the cars
for (x, y, w, h) in cars:
    # face_coordinates is an array itself so [0] is to select the first one or the first face
    # (x, y, w, h) = face_coordinates[0]
    # the last two parts are for the color (Blue-Green-Red, thickness of the rectangle)
    cv2.rectangle(img,  (x, y), (x+w, y+h),
                  (0, randrange(255), 0, 2))  # rand range is a function being used to make sure different colors pop up every time for the face detection
# display the image with the cars spotted
cv2.imshow('Hamza AI detection app', img)

# don't auto-close (wait in the opened window and listen for a key press)
cv2.waitKey()
