import cv2
from random import randrange

video = cv2.VideoCapture('')

# Pre-trained car classifier
classifier_file = 'car_detector.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# iterate forever over frames until the video ends
while True:

    # read the current frame
    (successful_frame_read, frame) = video.read()

    if successful_frame_read:
        # Now make the image black and white for the algorithm to understand
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else:
        break

    # to quit the app by using the key "Q", the key is being fetched from the waitKey function and the ASCII characters are being compared
    if key == 81 or key == 113:
        break

    # detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    # drawing the rectangles around the face to easily identify the faces
    # looping through the data if there are more faces
    for (x, y, w, h) in cars:
        # face_coordinates is an array itself so [0] is to select the first one or the first face
        # (x, y, w, h) = face_coordinates[0]
        # the last two parts are for the color (Blue-Green-Red, thickness of the rectangle)
        cv2.rectangle(frame,  (x, y), (x+w, y+h),
                      (0, 255, 0), 2)  # rand range is a function being used to make sure different colors pop up every time for the face detection

    # display the frame with the car spotted
    cv2.imshow("Car Detector", frame)

    # In python this command is used to keep the window open until a key is pressed to clear it otherwise the window quickly shows up and closes, it is hard to notice.
    # using the variable key to store what key is being pressed and then using it later on
    key = cv2.waitKey(1)

# to release the videocapture object
video.release()
