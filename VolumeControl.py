import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#------------------------------------------
widthCam, heightCam = 640, 480 # Adjust the area screen of the camera
#------------------------------------------

cap = cv2.VideoCapture(0)
# Set the width and height of the camera
cap.set(3, widthCam)
cap.set(4, widthCam)
# Set previous time for calculating fps
pTime = 0

# Create a hand detector object
detector = htm.handDetector(detectionCon=0.7)

"""
To modify with the audio of the computer, we use pycaw.
Source: https://github.com/AndreMiras/pycaw
"""
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img, True)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # Check MediaPipe's documentation for the position of each part of the hand
        # Link: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
        # print(lmList[4], lmList[8])

        # Get the coordinates of the thumb and index fingers.
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        # Center coordinates between two lines
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Draw a circle on the tip of the thumb and index fingers.
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        # Draw a line between the tip of the thumb and index fingers.
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        # Draw a circle between two lines.
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # Get the length between two points
        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        # Hand range 50 - 300
        # Volume range -65 - 0
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # Create a volume bar on the side of the camera
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 2)



    # Calculate the fps
    cTime = time.time()
    fps = 1 /(cTime - pTime)
    pTime = cTime

    # Put the fps number on the screen. Parameter: (what to put, coordinates, font, scale, color, thickness)
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Monitor", img)
    cv2.waitKey(1)