"""
python face_detect.py ../../ami/amicorpus/ES2014a/video/ES2014a.Closeup1_images/  ./haarcascade_frontalface_default.xml
"""

import sys
sys.path.insert(1,'/usr/local/opencv/2.4.5/lib64/python2.7')
import cv2
import glob
# import ipdb
import os

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image_paths = glob.glob(os.path.join(imagePath, '*.jpg'))
# print image_paths[:10]
# ipdb.set_trace()
for image_path in image_paths:
    # print image_path
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    print "Found {} faces!".format(len(faces))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # cv2.imshow("Faces found", image)
    # cv2.waitKey(0)











