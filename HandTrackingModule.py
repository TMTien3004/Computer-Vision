import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5): 
        self.mode = mode # False by default
        self.maxHands = maxHands # 2 hands by default
        self.detectionCon = detectionCon # 0.5 by default
        self.trackCon = trackCon # 0.5 by default

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1,
                                        self.detectionCon, self.trackCon) # We use the default input parameter of the Hands() function.
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]
 
    # Create a function to find the hands
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks: # Check if there is anything on the screen
            for handLms in self.results.multi_hand_landmarks:
                # If we want to draw our hand (we set it as True by default), we use the draw_landmarks() function
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    # Track the position of the hand
    def findPosition(self, img, handNo=0, draw=True):

        self.lmList = []
        if self.results.multi_hand_landmarks:  # Check if there is anything on the screen
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape # Find the height, width and channels of the image
                cx, cy = int(lm.x * w), int(lm.y * h) # Find position of the image
                # print(id, cx, cy) # Print the position of the image
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0 ,255), cv2.FILLED)

        return self.lmList

    # Count how many fingers are raised
    def fingersUp(self):
        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # The rest of 4 fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():
    pTime = 0
    cTime = 0

    # Create video object
    cap = cv2.VideoCapture(0) # Use Webcam number 0

    detector = handDetector()

    while True:
        success, img = cap.read()

        # Find the hands
        img = detector.findHands(img)

        # Find the position of the hands
        lmList = detector.findPosition(img)
        # Here, you can print out the position of certain parts of your finger.
        # For example, 0 is your wrist, 4 is the tip of your thumb,...
        if len(lmList) != 0:
            print(lmList[8])

        # Display FPS
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2) # Display FPS text

        cv2.imshow("Monitor", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()