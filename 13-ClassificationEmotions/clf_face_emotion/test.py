import numpy as np
import cv2
import pickle
import dlib
import os

def load_model(inp_name):
    with open(inp_name, 'rb') as f:
        out_clf = pickle.load(f)
        return out_clf

if __name__ == "__main__":
    
    class_name =  ["surprised", "angry", "happy", "sad", "neutral"]
    model_im = load_model("models/random_forest_im.pkl")
    model_points = load_model("models/random_forest_points.pkl")
    #model_points = load_model("models/svm_points.pkl")
    
    
    cap = cv2.VideoCapture(0)
    
    # Init dlib
    detector = dlib.get_frontal_face_detector()
    path_model = os.path.join("models", "shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor(path_model)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
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
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

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

            # Case images
            # preprocessing
            mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
            im = mask_gray/255.0
            im = cv2.resize(im, (128,128), interpolation = cv2.INTER_AREA)
            im = im.reshape(128*128)
            res_im = model_im.predict([im])
            print("----------------image res: ", class_name[int(res_im)])
    
            # Case vector points
            # preprocessing
            points_x = points[:,0]
            points_y = points[:,1]
            points_x = (points_x - points_x.min()) / (points_x.max() - points_x.min())
            points_y = (points_y - points_y.min()) / (points_y.max() - points_y.min())
            points = np.asarray([points_x,points_y]).reshape(68*2)
            #points = (points - points.min()) / (points.max() - points.min())
            #print(points)
            res_points = model_points.predict([points])
            #print(res_points)
            print("points res: ", class_name[int(res_points)])
            
        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()