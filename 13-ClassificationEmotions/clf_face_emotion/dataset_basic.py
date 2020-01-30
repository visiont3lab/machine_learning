# Opencv Video Capture : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
import numpy as np
import cv2
import keyboard
import os

if __name__ == "__main__":
    
    # Init variable
    class_name =  ["surprised", "angry", "happy", "sad"]
    num_classes = len(class_name)
    folder_path = "dataset/"
    for i in range(0,num_classes):
        try: 
            path = os.path.join(".", folder_path + class_name[i])
            os.makedirs(path, exist_ok = True) 
        except OSError as error:    
            print("Directory '%s' can not be created") 
  
    
    # Select a class
    selected_class = class_name[0]
    count = 0   
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our algorithm
        name = os.path.join(folder_path,selected_class)
        name = os.path.join(name,str(count) + ".png")
        print(name)
        
        cv2.imshow("Image",frame)
        
        k = cv2.waitKey(33)
        if k == ord('s'):
            print("Saving mask")
            cv2.imwrite(name, frame) 
            count = count + 1

        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()