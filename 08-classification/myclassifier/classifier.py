from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
import cv2
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from keras.preprocessing.image import ImageDataGenerator    

def save_model(inp_name,inp_clf):
    #https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn
    with open(inp_name, 'wb') as f:
        pickle.dump(inp_clf, f) 

def load_model(inp_name):
    with open(inp_name, 'rb') as f:
        out_clf = pickle.load(f)
        return out_clf

def classfier_analysis(clf, X_train, Y_train, X_test, Y_test):
 
    threshold_precision = 0.9

    Y_pred = cross_val_predict(clf, X_train,Y_train, cv=3)
    # Random forest
    Y_probas = cross_val_predict(clf, X_train, Y_train, cv=3,method="predict_proba")
    # SGD
    Y_scores = cross_val_predict(clf, X_train,Y_train, cv=3, method="decision_function")

    Y_scores = Y_probas[:, 1] # score = proba of positive class
    precisions, recalls, thresholds = precision_recall_curve(Y_train, Y_scores)
    threshold_desired_precision = thresholds[np.argmax(precisions >= threshold_precision)] # 90%

    Y_pred = (Y_scores >= threshold_desired_precision)

    conf_matrix = confusion_matrix(Y_train, Y_pred)
    print("Confusion matrix Train Set: ")
    print(conf_matrix)
    res_precision_score = precision_score(Y_train, Y_pred) 
    res_recall_score = recall_score(Y_train, Y_pred) 
    res_f1_score = f1_score(Y_train, Y_pred)
    print("prediction: ", np.round(res_precision_score,2), "recall: ", np.round(res_recall_score,2), "f1_score: ", np.round(res_f1_score,2))

    fpr, tpr, thresholds_roc = roc_curve(Y_train, Y_scores)
    roc_auc = roc_auc_score(Y_train, Y_scores)
    print("ROC AUC score: ", roc_auc)

    plt.subplot(3, 1, 1)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend()   
    plt.plot(threshold_desired_precision, res_precision_score, 'bo')
    plt.plot(threshold_desired_precision, res_recall_score, 'go')
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(recalls[:-1],precisions[:-1])
    plt.plot(res_recall_score,res_precision_score,'ro')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(fpr, tpr, "b--", linewidth=2, label="binary_classifier")
    plt.plot([0, 1], [0, 1], 'k--', label="random_classifier") 
    plt.plot(res_recall_score,'bo')
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR) recall")
    plt.grid()
    plt.legend()
    plt.show()


class MyDataset():
    
    def load_dataset(self):
        print("Loading Dataset")
        mnist = fetch_openml('mnist_784', version=1)
        X = mnist = fetch_openml('mnist_784', version=1)
        X = mnist["data"]/255
        Y = mnist["target"]
        print(mnist.keys())
        Y = Y.astype(np.uint8) # convert string target to int
        #X_train, X_test, Y_train, Y_test = X[:60000],X[60000:], Y[:60000], Y[60000:]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
        # override
        #Y_train = (Y_train == 7) # True for all 5s, False for all other digits.
        #Y_test = (Y_test == 7)
        print("X_train shape: ", X_train.shape)
        print("Y_train shape: ", Y_train.shape)
        print("X_test shape: ", X_test.shape)
        print("Y_test shape: ", Y_test.shape)

        X_train, Y_train = self.data_augmentation(X_train, Y_train)

        return X_train, Y_train, X_test, Y_test

    def data_augmentation(self, X_train, Y_train):
        # https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
        # https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        num_train_data = X_train.shape[0]
        X_train = np.array([X_train.reshape((num_train_data,28,28)), np.newaxis])
        batch_size = num_train_data + 3000
        
        datagen.fit(X_train)
        X_train, Y_train = datagen.flow(X_train,Y_train, batch_size)
        X_train = X_train.reshape((batch_size,28*28))
        return X_train, Y_train

'''
class MyCNNClassifier():
    
    def __init__(self):

    def get_model(self):
        model=models.Sequential()
        model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(150, 150,1))) 
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(4, activation='softmax'))
        return model

    def train(self):

        batch_size=64
        num_classes=4
        epochs=5

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2) #0.5,
            #rotation_range=20,
            #horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            'datasets/GestureRecognition/train',
            color_mode="grayscale",
            shuffle=True,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            'datasets/GestureRecognition/val',
            color_mode="grayscale",
            shuffle=False,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='categorical')

        checkpoint = ModelCheckpoint(
            './base_model.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            save_weights_only=True,
            period=1
        )
        earlystop = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=30,
            verbose=1,
            mode='auto'
        )
        reduce = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3,
            verbose=1, 
            mode='auto'
        )

        callbacks = [checkpoint,reduce]
        model = get_model()
        model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
        print(model.summary())
        History = model.fit_generator(generator=train_generator, validation_data=validation_generator, steps_per_epoch=100, validation_steps=20, epochs=epochs,  verbose=1,callbacks=callbacks)
'''

class MyRandomForestClassifier():
    
    def __init__(self):
        print("Using Random Forest Classifier")
        self.clf = RandomForestClassifier(n_estimators=10,random_state=42)
        #self.clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
        #            solver='sgd', verbose=10, random_state=1,
        #            learning_rate_init=.1)
    
    def train(self, X_train, Y_train ):
        self.clf.fit(X_train, Y_train)
        #Y_pred = self.clf.predict([X_test[4]])
        train_acc = cross_val_score(self.clf, X_train, Y_train, cv=3, scoring="accuracy")
        print("accuracy train set split in 3 set: ", np.round(train_acc,2))
        
        save_model("random_forest_classifier.pkl",self.clf)

        return self.clf

class MySGClassifier():
    
    def __init__(self):
        print("Using SGD Classifier")
        self.clf = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
            eta0=0.0, fit_intercept=True, l1_ratio=0.15,
            learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
            penalty='l2', power_t=0.5, random_state=None, shuffle=True,
            verbose=0, warm_start=False)   
    
    def train(self, X_train, Y_train ):
        self.clf.fit(X_train, Y_train)
        #Y_pred = self.clf.predict([X_test[4]])
        train_acc = cross_val_score(self.clf, X_train, Y_train, cv=3, scoring="accuracy")
        print("accuracy train set split in 3 set: ", np.round(train_acc,2))
        Y_pred = cross_val_predict(self.clf, X_train,Y_train, cv=3)
        
        return self.clf

class MyCameraCapture():
   
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        data = MyDataset()
        X_train, Y_train, X_test, Y_test = data.load_dataset()
        #inp = X_train[0].reshape(28, 28)

        #cv2.imshow("Test image",inp)
        #cv2.waitKey(0)
        #print(inp)
        
        # Train
        #clf = MySGClassifier()
        clf = MyRandomForestClassifier()
        self.my_clf = clf.train(X_train, Y_train)
        
        # Test
        #self.my_clf = load_model("random_forest_classifier.pkl")
        
    def run(self):
        print("Starting Real time Test")
        while(True):
            ret, frame = self.cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 21)
            #gray = cv2.bilateralFilter(gray,9,75,75)
            thresh = 100
            maxValue = 255
            th, dst = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_BINARY);
            dst = cv2.bitwise_not(dst)
            gray = dst/255
            resized_img = np.asarray(cv2.resize(gray, (28, 28)))
            inp_img = resized_img.flatten()
            Y_pred = self.my_clf.predict_proba([inp_img])[0]
            ind = np.argmax([self.my_clf.predict_proba([inp_img])])
            if (Y_pred[ind]>0.20):
                print(ind)
                print(Y_pred[ind])
                cv2.putText(frame, str(ind),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),3) 
            cv2.imshow('frame',frame)
            cv2.imshow("threshold", resized_img*255)
            cv2.waitKey(1)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    
    MC = MyCameraCapture()
    MC.run()

