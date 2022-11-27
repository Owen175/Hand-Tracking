from typing import Type

import cv2
import time
import mediapipe as mp
import pyautogui
import mouse
import hand_tracking_module as htm
from hand_tracking_module import handDetector


def moveMouse(screen_width, screen_height, lm):
    mouse.move(lm.x*screen_width, lm.y*screen_height)


screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

hand_detector = htm.handDetector()

pTime = 0
cTime = 0

while True:
    success, input_img = cap.read()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    img = hand_detector.findHands(img=input_img, draw=True, finger=8)
    #if hand_detector.fingerPos != None:
    #    moveMouse(screen_width, screen_height, hand_detector.fingerPos)
    #    hand_detector.fingerPos = None
    cv2.putText(img, str(int(fps)), (10,70),
        cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
