import cv2
import mediapipe as mp
import time
import math

class handdetector():
    def __int__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.tip =[4, 8, 12, 16, 20]
    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        if self.result.multi_hand_landmarks:
            for halm in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, halm, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handno=0, draw=True):
        self.lmlist = []
        xlist = []
        ylist = []
        bbox =[]
        if self.result.multi_hand_landmarks:
            mHand = self.result.multi_hand_landmarks[handno]
            for id, lm in enumerate(mHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xlist.append(cx)
                ylist.append(cy)
                self.lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 20, (0, 255, 0), cv2.FILLED)
            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img,(xmin-20,ymin-20),(xmax+20,ymax+20),(0, 255, 0), 2)
        return self.lmlist, bbox
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x, y = self.lmlist[4][1], self.lmlist[4][2]
        x1, y1 = self.lmlist[8][1], self.lmlist[8][2]
        cx, cy = (x + x1) // 2, (y + y1) // 2
        if draw:
            cv2.line(img, (x, y), (x1, y1), (255, 0, 0), t)
            cv2.circle(img, (x, y), r, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), r, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x1 - x, y1 - y)
        return length, img, [x,y,x1,y1,cx,cy]
    def fingersUp(self):
        fingers = []
        # thumb
        if self.lmlist[self.tip[0]][1] > self.lmlist[self.tip[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if self.lmlist[self.tip[id]][2] < self.lmlist[self.tip[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handdetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist)!=0:
            print(lmlist[4])
        cv2.resize(img, [500, 500])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
