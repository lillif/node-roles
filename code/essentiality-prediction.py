import numpy as np
import matplotlib.pyplot as plt
import pickle

# importing models from sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# sns for the confusion matrix plot
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.metrics import precision_recall_curve, f1_score
from itertools import cycle
# importing sys for exiting when necessary
import sys

from scipy import interp
from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
from sklearn.metrics import roc_auc_score

# importing sys for exiting when necessary
import sys

# prediction from rolx
def rolx_training_data():
    fpath = '../data/rolx-train-val-test/'
    X = np.load(f'{fpath}Xtrain.npy')
    y = np.load(f'{fpath}ytrain.npy')
    yc = np.load(f'{fpath}yctrain.npy')
    return X, y, yc

# prediction from flow profiles
def flow_profiles_training_data(model):
    simpath = '../data/role-based-sim-features/'
    rolxpath = '../data/rolx-memberships/'
    X = np.load(f'{simpath}{model}-X.npy')
    Y = np.load(f'{simpath}{model}-Y.npy')
    y = np.load(f'{rolxpath}{model}-y-num.npy')
    yc = np.load(f'{rolxpath}{model}-y-classes.npy')
    print(f'{model} has X shape {X.shape}')
    return X, Y, y, yc


def oversample(X, y):
    X0 = X[y==0]
    X1 = X[y==1]
    X2 = X[y==2]
    X3 = X[y==3]
    X1 = np.repeat(X[y==1], int(len(X0) / len(X1)), axis=0)
    X2 = np.repeat(X[y==2], int(len(X0) / len(X2)), axis=0)
    X3 = np.repeat(X[y==3], int(len(X0) / len(X3)), axis=0)
    d = max(len(X0) - len(X3), 0)
    X3 = np.hstack((X3.T, X3[0:d].T)).T

    X = np.hstack((X0.T, X1.T, X2.T, X3.T)).T

    y0 = np.repeat(0,len(X0))
    y1 = np.repeat(1,len(X1))
    y2 = np.repeat(2,len(X2))
    y3 = np.repeat(3,len(X3))

    y = np.hstack((y0,y1,y2,y3))
    return X, y

def undersample(X, y):
    X0 = X[y==0]
    X1 = X[y==1]
    X2 = X[y==2]
    X3 = X[y==3]

    n = len(X2)

    x0s = np.random.RandomState(seed=134).randint(0,len(X0),size=n)
    x1s = np.random.RandomState(seed=723).randint(0,len(X1),size=n)
    x3s = np.random.RandomState(seed=941).randint(0,len(X3),size=n)

    X0 = X0[x0s]
    X1 = X1[x1s]
    X3 = X3[x3s]

    y0 = np.repeat(0,len(X0))
    y1 = np.repeat(1,len(X1))
    y2 = np.repeat(2,len(X2))
    y3 = np.repeat(3,len(X3))

    X = np.hstack((X0.T, X1.T, X2.T, X3.T)).T
    y = np.hstack((y0,y1,y2,y3))

    return X, y

def plotconfmatrix(yv, yp, clf):
    # plot confusion matrix
    # Get and reshape confusion matrix data
    matrix = confusion_matrix(yv, yp)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    
    # Build the plot
    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2)
    
    # Add labels to the plot
    class_names = ['No Change', 'Mild Change', 'Severe Change', 'Lethal']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for ' + str(clf).split('(')[0])
    plt.show()


def roccurve(classifier, X_train, y_train, X_test, y_test, n_classes = 4):
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[i], y_score[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute ROC curve and ROC area for each class
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
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def training(X, y, clf, K = 5, _plot = False, oversmpl = False):

    p = np.random.RandomState(seed=847).permutation(len(y))
    Xsh = X[p]
    ysh = y[p]

    print(len(Xsh))
    print(len(ysh))

    splits = [int(i * len(y) / K) for i in range(K+1)]


    errs = []
    acc = []

    for k in range(K):
        
        Xt = np.concatenate((Xsh[:splits[k]], Xsh[splits[k+1]:]))
        yt = np.concatenate((ysh[:splits[k]], ysh[splits[k+1]:]))
        
        if oversmpl:
            Xt, yt = oversample(Xt, yt)
        
        Xv = Xsh[splits[k]:splits[k+1]]
        yv = ysh[splits[k]:splits[k+1]]

        clf.fit(Xt, yt)

        yp = clf.predict(Xv)
        
        acc.append(accuracy_score(yv, yp))
        
        wrong = np.ones_like(yp)
        wrong[yp == yv] = 0
        errs.append(sum(wrong) / len(yv))
        
        if _plot:
            # plot ROC curve
            # roccurve(clf, Xt, yt, Xv, yv)
            # plot confmatrix
            plotconfmatrix(yv, yp, clf)
            
    avgerr = sum(errs) / len(errs)
    avgacc = sum(acc) / len(acc)
    
    return avgerr, avgacc

def train_and_print(X, y, clf, featureset, _plot = False, oversmpl = True):
    avgerr, avgacc = training(X, y, clf, _plot = _plot, oversmpl = oversmpl)
    print(f'{str(clf).split("(")[0]} classifier on {featureset}:')
    print(f'average cross-validation error = {avgerr}, average cross-validation accuracy = {avgacc}', end='\n\n')


# rolX predictions
Xr, yr, ycr = rolx_training_data()

# # random forests:
# for d in range(1,25):
#     forest = RandomForestClassifier(max_depth=d, random_state=0)
#     train_and_print(Xr, ycr, forest, f'rolx roles with depth {d}', _plot = True, oversmpl = True)
#     break

# # multi-layer perceptron:
# mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# train_and_print(Xr, ycr, mlp, 'rolx roles', _plot = True, oversmpl = True)


# support vector machine:
svm = svm.SVC(decision_function_shape='ovo', probability=True) # svm.SVC(decision_function_shape='ovo')
train_and_print(Xr, ycr, svm, 'rolx roles', _plot = True, oversmpl = True)

# logistic regression:
logreg = LogisticRegression(random_state=0)
train_and_print(Xr, ycr, logreg, 'rolx roles', _plot = True, oversmpl = True)

# models = ['BT-549', 'HCT-116','K-562', 'MCF7', 'OVCAR-5']

# # flow profiles redictions
# Xf, Yf, yf, ycf = flow_profiles_training_data()

# Xo, yco = oversample(Xf, ycf)
# Xu, ycu = undersample(Xf, ycf, ncps = 3)


