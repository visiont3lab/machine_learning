import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.widgets import Slider, Button, RadioButtons
from sklearn.model_selection import train_test_split
from utils import save_model
import os
import cv2
from sklearn.metrics import confusion_matrix, auc,precision_score,recall_score
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import interp
from itertools import cycle
from sklearn.naive_bayes import GaussianNB 
from sklearn.multiclass import OneVsRestClassifier
import pickle

def roc_analysis_full(Y_test,Y_scores,n_classes):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    # Compute ROC curve and ROC area for each class
    Y_test = label_binarize(Y_test, classes=[*range(n_classes)]) # make categorical
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.4f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.4f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.4f})'
                ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

def prediction_recall_full(Y_test,Y_scores,n_classes):
    Y_test = label_binarize(Y_test, classes=[*range(n_classes)]) # make categorical
    # https://stackoverflow.com/questions/56090541/how-to-plot-precision-and-recall-of-multiclass-classifier
    precision = dict()
    recall = dict()
    threshold = dict() 
    fig = plt.figure()
    for i in range(n_classes):
        precision[i], recall[i], threshold[i] = precision_recall_curve(Y_test[:, i], Y_scores[:, i])
        #print(threshold[i].shape)
        #print(recall[i][:-1].shape)
        ax = fig.add_subplot(3,2,i+1)
        ax.plot(threshold[i], precision[i][:-1], 'b--', label='precision ' + "class_" + str(i))
        ax.plot(threshold[i], recall[i][:-1],'g-', label='recall')
        ax.legend()
        ax.grid()

    #plt.xlabel("Probabilities Threshold")
    #plt.ylabel("Percentage")
    plt.show()

def precision_vs_recall(Y_test,Y_scores,n_classes):
    Y_test = label_binarize(Y_test, classes=[*range(n_classes)]) # make categorical
    # precision recall curve
    precision = dict()
    recall = dict()
    for i in range(0,n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],Y_scores[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()

def get_thresholds(Y_test,Y_scores,n_classes, precision_vec):
    Y_test = label_binarize(Y_test, classes=[*range(n_classes)]) # make categorical
    precision = dict()
    recall = dict()
    threshold = dict() 
    thresholds_chosen = np.zeros(n_classes)
    for i in range(n_classes):
        precision[i], recall[i], threshold[i] = precision_recall_curve(Y_test[:, i], Y_scores[:, i])
        precision_chosen = precision_vec[i]
        threshold_chosen = threshold[i][np.argmax(precision[i] >= precision_chosen)]
        thresholds_chosen[i]=threshold_chosen
    return thresholds_chosen

def save_model(inp_name,inp_clf):
    #https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn
    with open(inp_name, 'wb') as f:
        pickle.dump(inp_clf, f) 

def load_model(inp_name):
    with open(inp_name, 'rb') as f:
        out_clf = pickle.load(f)
        return out_clf
