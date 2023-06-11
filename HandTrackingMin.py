import cv2
import mediapipe as mp
import time

# Create video object
cap = cv2.VideoCapture(0) # Use Webcam number 0

mpHands = mp.solutions.hands
hands = mpHands.Hands() # We use the default input parameter of the Hands() function.
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks: # Check if there is anything on the screen
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape # Find the height, width and channels of the image
                cx, cy = int(lm.x * w), int(lm.y * h) # Find position of the image
                print(id, cx, cy)

                if id == 4:
                    cv2.circle(img, (cx, cy), 25, (255, 0 ,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Display FPS
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2) # Display FPS text

    cv2.imshow("Monitor", img)
    cv2.waitKey(1)