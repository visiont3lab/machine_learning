import numpy as np
import cv2
from utils import load_model

def hsv_filter(frame):
    # hasv color space
    frame = cv2.GaussianBlur(frame,(5,5),30) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_skin = np.array([0,18,36], dtype=np.uint8)
    upper_skin = np.array([25,229,243], dtype=np.uint8)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
 
    # Contour extraction
    contours,hierarchy= cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    mask_fin = np.zeros(mask.shape, np.uint8)
    if len(contours)!=0:
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
       
        # Draw contour and hull       
        hull = cv2.convexHull(cnt)
        cv2.drawContours(res, [cnt], 0, (0, 255, 0), 2)
        cv2.drawContours(res, [hull], 0, (0, 0, 255), 3)
        cv2.drawContours(mask_fin, [cnt], -1, (255),cv2.FILLED)

    return res, mask_fin

def kmeans_filter(img):
    # convert to np.float32
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2 # number of clusters
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def genererate_mask(frame):
    '''
    1. Convert Gray Image
    2. Threshold to only have interested color
    3. Find largest contour
    4. Generate the mask
    5. Save the obtained mask
    '''

    # Flip image
    #frame = cv2.flip(frame,1)

    #print("Original Image Shape: ", frame.shape)

    # Generate region of interest
    roi=frame[50:320,415:630]
    cv2.rectangle(frame,(415,50),(630,320),(0,255,0),0)

    # Create gray image
    #im = roi # cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # get current positions of four trackbars
    #im = kmeans_filter(roi)
    im, mask = hsv_filter(roi)

    cv2.imshow("Frame", frame)
    cv2.imshow("Image", im)
    cv2.imshow("Mask", mask)
    #.waitKey(0)    
    return mask  

if __name__ == "__main__":
    
    #model = load_model("knn.pkl")
    model = load_model("random_forest.pkl")
    
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our algorithm
        mask = genererate_mask(frame)
        mask = cv2.resize(mask, (256,256), interpolation = cv2.INTER_AREA)
        mask = mask.reshape(256*256)
        mask = mask/255.0
        res = model.predict([mask])
        print(res)


        k = cv2.waitKey(33)
        if k == ord('s'):
            print("Saving mask")
            count = count + 1

        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()