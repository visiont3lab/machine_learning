# Opencv Video Capture : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
import numpy as np
import cv2
import keyboard
import os
import dlib
import json

if __name__ == "__main__":
    
    # Init variable
    class_name =  ["surprised", "angry", "happy", "sad", "neutral"]
    num_classes = len(class_name)
    folder_path = "dataset/"
    for i in range(0,num_classes):
        try: 
            path = os.path.join(".", folder_path + class_name[i])
            os.makedirs(path, exist_ok = True) 
        except OSError as error:    
            print("Directory '%s' can not be created") 
  
    # Select a class
    selected_class = class_name[4]  # change this
    count = 0   
    cap = cv2.VideoCapture(0)

    # Init dlib
    detector = dlib.get_frontal_face_detector()
    path_model = os.path.join("models", "shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor(path_model)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our algorithm
        name = os.path.join(folder_path,selected_class)
        name = os.path.join(name,str(count)) # + ".png")

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(img_gray)
        faces = detector(img_gray)
        for face in faces:
            landmarks = predictor(img_gray, face)
            landmarks_points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))
                #cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            points = np.array(landmarks_points, np.int32)
            # Convex HULL
            convexhull = cv2.convexHull(points)
            #cv2.polylines(frame, [convexhull], True, (255, 0, 0), 3)
            #cv2.fillConvexPoly(mask, convexhull, 255)
            #frame = cv2.bitwise_and(frame, frame, mask=mask)
            # rectangle 
            rect = cv2.boundingRect(convexhull)
            (x,y,w,h) = rect
            #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0))
            mask[y:y+h,x:x+w] = 255
            frame = cv2.bitwise_and(frame, frame, mask=mask)
            mask_frame = frame[y:y+h,x:x+w]
            
            cv2.imshow("Image",mask_frame)
            
            k = cv2.waitKey(33)
            if k == ord('s'):
                print("Saving mask")

                with open(name + '.json', 'w') as outfile:
                    json_data = {'data' : points.tolist()} 
                    json.dump(json_data, outfile)
                '''
                with open(name+".json") as json_file:
                    data = json.load(json_file)
                    #print(data["data"]-points)
                '''
                cv2.imwrite(name + ".png", mask_frame) 
                count = count + 1

        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()