
import numpy as np
import pickle
from PIL import Image
import cv2
with open('/home/schneider/Desktop/ML/Assignment3/task1/Ridge_Model.pkl', 'rb') as f:
    RM = pickle.load(f)


def imgreshape(frame):
    new_image=[]
    new_size = (32,32,3)
    resized_img = np.resize(frame,new_size)
    img_ = np.divide(resized_img, 255)
    img = img_.reshape(3072)
    new_image.append(img)
    return new_image

import cv2

# Open the default camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error opening video stream")
    
# Read until the video is completed or user quits
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        resized_frame = imgreshape(frame)
        predictLR = int(RM.predict(resized_frame))

        # Display the resulting frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 
                str(predictLR), 
                (400, 300), 
                font, 1, 
                (255, 0, 0), 
                1, 
                cv2.LINE_4)
        cv2.imshow('Frame',frame)
        key = cv2.waitKey(25)
        if key == (27) or key == ord('q'):
            break
    else:
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()


