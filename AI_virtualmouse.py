import cv2
import numpy as np
import handtrackingmodule as htm
import time
import autopy

wcam, hcam = 640, 480
cap =cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4,480)
ptime = 0
frameR =100
smooth = 5

prevLocx,prevLocy = 0,0
currLocx, currLocy =0,0
wScr, hScr = autopy.screen.size()
detector = htm.handdetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img, draw = False)
    if len(lmlist)!=0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        fingers =detector.fingersUp()
        cv2.rectangle(img, (frameR, frameR), (wcam - frameR, hcam - frameR), (255, 0, 0), 2)
        if fingers[1]==1 and fingers[2]==0:

            x = np.interp(x1,(frameR,wcam-frameR),(0,wScr))
            y = np.interp(x1, (frameR, hcam-frameR), (0, hScr))
            # to smoothen the values
            currLocx= prevLocx+(x-prevLocx)/smooth
            currLocy=prevLocy+(y-prevLocy)/smooth
            

            autopy.mouse.move(wScr-currLocx,currLocy)
            cv2.circle(img,(x1,y1),15,(0,255,0),cv2.FILLED)
            prevLocx, prevLocy =currLocx,currLocy
        if fingers[1] == 1 and fingers[2] == 1:
            length, lineInfo = detector.findDistance(8,12,img)
            if length< 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (255, 0, 0), cv2.FILLED)
                autopy.mouse.click()

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)
