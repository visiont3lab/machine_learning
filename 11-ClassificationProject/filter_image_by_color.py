import cv2
import numpy as np

def nothing(x):
    pass

# Pink r=255 g=192 b=203

# Create a black image, a window
cv2.namedWindow('image')
 
# create trackbars for color change
cv2.createTrackbar('lh','image',0,255,nothing)
cv2.createTrackbar('ls','image',36,255,nothing)
cv2.createTrackbar('lv','image',18,255,nothing)
cv2.createTrackbar('hh','image',25,255,nothing)
cv2.createTrackbar('hs','image',229,255,nothing)
cv2.createTrackbar('hv','image',243,255,nothing)
  
cap = cv2.VideoCapture(0)

while(1):
    
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    roi=frame[50:320,415:630]
    cv2.rectangle(frame,(415,50),(630,320),(0,255,0),0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # get current positions of four trackbars
    lh = cv2.getTrackbarPos('lh','image')
    ls = cv2.getTrackbarPos('ls','image')
    lv = cv2.getTrackbarPos('lv','image')

    hh = cv2.getTrackbarPos('hh','image')
    hs = cv2.getTrackbarPos('hs','image')
    hv = cv2.getTrackbarPos('hv','image')
  
    lower_skin = np.array([lh,ls,lv], dtype=np.uint8)
    upper_skin = np.array([hh,hs,hv], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    im = cv2.bitwise_and(roi,roi, mask= mask)


    print(upper_skin, lower_skin)
  
    cv2.imshow('image',im)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break