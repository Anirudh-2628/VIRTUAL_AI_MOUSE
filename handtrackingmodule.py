import cv2
import mediapipe as mp
import time
import autopy
import numpy as np


class handdetector():
    def __init__(self, maxHands=1, mode=False, detectionCon=0.5, tracCon=0.5):
        self.mode = mode
        self.maxhands = maxHands
        self.detectionCon = detectionCon
        self.tracCon = tracCon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            self.mode, self.maxhands, self.detectionCon, self.tracCon)
        self.mpdraw = mp.solutions.drawing_utils
    
    def findhands(self, img, draw=True):       
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, hand_landmarks, self.mphands.HAND_CONNECTIONS)
        return img


    def findposition(self, img, handno=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
        return lmlist


    def finddist(lmlist):
        dist = lmlist[8][2] - lmlist[12][2] 
        return dist
    
        

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
wscr, hscr = autopy.screen.size()
wCam, hCam = 640, 480
plocx, plocy = 0, 0
clocx, clocy = 0, 0


cap.set(3, wCam)
cap.set(4, hCam)
ptime = 0
ctime = 0
framR = 100
smthindex = 5
detector = handdetector()
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findhands(img)
    lmlist = detector.findposition(img)
    cv2.rectangle(img, (framR, framR), (wCam-framR, hCam-framR), (0,0,0), 2)

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]
        fingercount = 0
        dist = lmlist[8][2] - lmlist[12][2]
        if lmlist[8][2] < lmlist[6][2]:
            fingercount = 1
            if lmlist[12][2] < lmlist[10][2]:
                fingercount = 2

        if fingercount==1:
            
            x3 = np.interp(x1, (framR,wCam-framR), (0,wscr))
            y3 = np.interp(y1, (framR,hCam-framR), (0,hscr))
            clocx = plocx + (x3 - plocx)/smthindex
            clocy = plocy + (y3 - plocy)/smthindex

            autopy.mouse.move(wscr-(wscr - clocx), clocy)

            plocx, plocy = clocx, clocy

        if fingercount==2:
            if dist<=10:
                autopy.mouse.click()



    ctime = time.time() 
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10, 10),
            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
    cv2.imshow("output", img)

    cv2.waitKey(1)


main()