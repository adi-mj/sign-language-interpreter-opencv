import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np 

#specefics for the text to be dispalyed
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 100)
fontScale = 2
color = (255, 0, 0)
thickness = 5

def process_img(hand_img):

    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    
    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)
    
    # accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(static_image_mode=True,
    max_num_hands=2, min_detection_confidence=0.7)

    # Results
    output = hands.process(img_flip)

    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        #print(data)
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)
                        
        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return(clean)
    except:
        #return an array of zeros if no detection is captured
        return 0
        # return(np.zeros(63, dtype=int))

import pickle
# load model
with open('model2.pkl', 'rb') as f:
    svm = pickle.load(f)

import cv2 as cv
cap = cv.VideoCapture(0)

i = 0    
while True:
    
    ret, frame = cap.read()    
    data = process_img(frame)
    
    #on unsuccessfull detection
    if data == 0:
        frame = cv2.putText(frame, "", org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
    
    #on successfull detection
    else:
        data = np.array(data)
        y_pred = svm.predict(data.reshape(-1,63))
        print(y_pred)
        
        frame = cv2.putText(frame, str(y_pred), org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
    
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()