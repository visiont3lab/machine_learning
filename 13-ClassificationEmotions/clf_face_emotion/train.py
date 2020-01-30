import os
import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, auc,precision_score,recall_score
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def save_model(inp_name,inp_clf):
    #https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn
    with open(inp_name, 'wb') as f:
        pickle.dump(inp_clf, f) 

## -----------  Read Data
class_name =  ["surprised", "angry", "happy", "sad", "neutral"]
num_classes = len(class_name)
X = []
L = []
Y = []
for i in range(0,num_classes):
    folder_path = os.path.join("dataset/", class_name[i])
    for name in os.listdir(folder_path):
        
        # Load face image
        name_txt = name.split(".")[0] + ".json"
        name_png = name.split(".")[0] + ".png"
        path_read_im = os.path.join(folder_path, name_png)
        path_read_vec = os.path.join(folder_path, name_txt)

        im = cv2.imread(path_read_im,0)
        im = im/255.0
        im = cv2.resize(im, (64,64), interpolation = cv2.INTER_AREA)
        im = im.reshape(64*64)

        # Load landmarks keypoints vec
        points = None
        with open(path_read_vec) as json_file:
            data = json.load(json_file)
            points = np.asarray(data["data"])
            points_x = points[:,0]
            points_y = points[:,1]
            points_x = (points_x - points_x.min()) / (points_x.max() - points_x.min())
            points_y = (points_y - points_y.min()) / (points_y.max() - points_y.min())
            points = np.asarray([points_x,points_y]).reshape(68*2)
        
            # Standard scaler z = (x-mean)/std
            #points = (points-np.mean(points))/np.std(points) 
            #points = (points - points.min()) / (points.max() - points.min()) #0  1 range
            #print(points)
            #print(points.shape)
            #print(data["data"]-points)

        X.append(im)
        L.append(points)
        Y.append(i)

X = np.array(X, dtype="float32")
Y = np.array(Y, dtype="float32")
L = np.array(L, dtype="float32")

print(X.shape)
print(Y.shape)
print(L.shape)

X_train, X_test, Y_train, Y_test = train_test_split(L, Y, test_size=0.3, shuffle=True, random_state=42)
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=42)

## ----------- Classifier
clf = RandomForestClassifier() 
#name = "models/random_forest_im.pkl"
name = "models/random_forest_points.pkl"

'''
#name = "models/svm_im.pkl"
name = "models/svm_points.pkl"
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
clf = GridSearchCV(
      SVC(probability=True), tuned_parameters, scoring='f1_macro'
)
'''

# Train analysis
clf.fit(X_train, Y_train)
accuracy_train = cross_val_score(clf, X_train, Y_train, cv=3, scoring="f1_weighted")
accuracy_test= cross_val_score(clf, X_test, Y_test, cv=3, scoring="f1_weighted")
print("f1_weighted score train set split in 3 set: ", accuracy_train)
print("f1_weighted score test set split in 3 set: ", accuracy_test)
#print(clf.best_estimator_)
#print(clf.best_score_)


# Test analysis
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
Y_proba = clf.predict_proba(X_test)
Y_scores = clf.predict(X_test)
precision_vec = precision_score(Y_test, Y_scores, average=None)
recall_vec = precision_score(Y_test, Y_scores, average=None)
cm = confusion_matrix(Y_test, Y_scores)
print("Precision Vector: ", precision_vec)
print("Recall Vector: ", recall_vec)
print("Confusion Matrix: ")
print(cm)

save_model(name, clf)