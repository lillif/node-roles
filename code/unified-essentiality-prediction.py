import numpy as np
import matplotlib as mpl
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

# importing sys for exiting when necessary
import sys

import pandas as pd

from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
from sklearn.metrics import roc_auc_score

# list of names of all five cell lines
models = ['BT-549', 'HCT-116', 'K-562', 'MCF7', 'OVCAR-5']


def essentialities():
    """
    Function to load and combine all essentiality values

    Returns
    -------
    e : np.array [N,]
        vector of essentiality values of all nodes in MFGs of all cell lines.

    """
    
    fpath = '../data/mfgs/'
    mdf = pd.read_csv(f'{fpath}{models[0]}_nodes.csv')
   
    e = np.array(mdf['essentiality'])
    for model in models[1:]:
        mdf = pd.read_csv(f'{fpath}{model}_nodes.csv')
        e = np.hstack([e, np.array(mdf['essentiality'])])
    
    return e

def binary_labels(e, t = 0.5):
    """
    Parameters
    ----------
    e : np.array (1-dimensional)
        essentiality values.
    t : float, optional
        threshold value for class 0/1. The default is 0.5.

    Returns
    -------
    c : np.array (1-dimensional)
        binary essentiality classes.

    """

    c = np.ones_like(e)
    c[np.round(e, decimals=10) <= t] = 0
    return c


def refx_features():
    """
    Function to load and combine all common RefX input features
    
    Returns
    -------
    X : np.array [N, D]
        feature matrix of common RefX features of all cell lines.

    """
    
    X = np.load(f'../data/rolx-features/{models[0]}-common-rolx-features-X.npy')
    for model in models[1:]:
        mpath = f'../data/rolx-features/{model}-common-rolx-features-X.npy'
        X = np.vstack([X, np.load(mpath)])
        
    return X

# prediction from flow profiles
def flow_profiles_training_data():
    """
    Function to load and combine all flow profiles.


    Returns
    -------
    X : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    yc : TYPE
        DESCRIPTION.

    """
    
    simpath = '../data/role-based-sim-features/'
    X = np.load(f'{simpath}{models[0]}-X.npy')
    for model in models[1:]:
        mpath = f'{simpath}{model}-X.npy'
        X = np.vstack([X, np.load(mpath)])
        
    return X


def oversample(X, y):
    """
    Function which copies samples from the smaller class to combat class imbalance.

    Parameters
    ----------
    X : np.array [N, D]
        Input feature matrix.
    y : np.array [N, ]
        Input class labels.

    Returns
    -------
    X : np.array [N, D']
        Input feature matrix with smaller class oversampled.
    y : TYPE
        DESCRIPTION.

    """
    X0 = X[y==0]
    X1 = X[y==1]
    
    diff = abs(len(X0) - len(X1))
    p = np.random.RandomState(seed=827).permutation(min(len(X0), len(X1)))
    
    if len(X0) > len(X1):
        print(f'if: x0: {len(X0)}, x1: {len(X1)}')
        X1 = np.repeat(X[y==1], int(len(X0) / len(X1)), axis=0)
        if len(X0) > len(X1):
            print(f'x0: {len(X0)}, x1: {len(X1)}')
            diff = len(X0) - len(X1)
            X1 = np.vstack((X1, X1[p[:diff]]))
        
    else:
        print(f'else: x0: {len(X0)}, x1: {len(X1)}')
        X0 = np.repeat(X[y==0], int(len(X1) / len(X0)), axis=0)
        if len(X1) > len(X0):
            print(f'x0: {len(X0)}, x1: {len(X1)}')
            diff = len(X1) - len(X0)
            X0 = np.vstack((X0, X0[p[:diff]]))
        
    
    X = np.vstack((X0, X1))

    y0 = np.repeat(0,len(X0))
    y1 = np.repeat(1,len(X1))

    y = np.hstack((y0,y1))
    return X, y


def plotconfmatrix(yv, yp, clf):
    # plot confusion matrix
    # Get and reshape confusion matrix data
    matrix = confusion_matrix(yv, yp)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    
    # Build the plot
    plt.figure()
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Blues, linewidths=0.2)
    
    # Add labels to the plot
    class_names = ['Non-Essential', 'Essential']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(str(clf).split('(')[0] + ' Confusion Matrix')
    plt.show()



def k_fold(X, y, clf, K = 5, _plot = False, oversmpl = False):

    p = np.random.RandomState(seed=847).permutation(len(y))
    Xsh = X[p]
    ysh = y[p]

    print(len(Xsh))
    print(len(ysh))
    
    print(f'sum y is {sum(y)}')

    splits = [int(i * len(y) / K) for i in range(K+1)]


    errs = []
    acc = []

    for k in range(K):
        
        Xt = np.concatenate((Xsh[:splits[k]], Xsh[splits[k+1]:]))
        yt = np.concatenate((ysh[:splits[k]], ysh[splits[k+1]:]))
        
        print((yt == 1).any())
        print((yt == 0).any())

        Xv = Xsh[splits[k]:splits[k+1]]
        yv = ysh[splits[k]:splits[k+1]]
        
        if oversmpl:
            Xt, yt = oversample(Xt, yt)
            # Xv, yv = oversample(Xv, yv)

        clf.fit(Xt, yt)

        yp = clf.predict(Xv)
        
        acc.append(accuracy_score(yv, yp))
        
        wrong = np.ones_like(yp)
        wrong[yp == yv] = 0
        errs.append(sum(wrong) / len(yv))
        
        if _plot:


            # plot ROC curve
            ns_probs = [0 for _ in range(len(yv))]
            lr_probs = clf.predict_proba(Xv)
            lr_probs = lr_probs[:, 1]
            ns_auc = roc_auc_score(yv, ns_probs)
            lr_auc = roc_auc_score(yv, lr_probs)
            # summarize scores
            print('No Skill: ROC AUC=%.3f' % (ns_auc))
            print('Logistic: ROC AUC=%.3f' % (lr_auc))
            # calculate roc curves
            ns_fpr, ns_tpr, _ = roc_curve(yv, ns_probs)
            lr_fpr, lr_tpr, _ = roc_curve(yv, lr_probs)
            # plot the roc curve for the model
            mpl.style.use('default')
            plt.figure()
            plt.plot(ns_fpr, ns_tpr, linestyle='--', color='#00325F', label='No Skill')
            plt.plot(lr_fpr, lr_tpr,  label= str(clf) + ' (area = %0.2f)' % lr_auc, color='#C10043')
            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # show the legend
            plt.legend(loc="lower right")
            plt.title('ROC Curve for ' + str(clf).split('(')[0])

            # show the plot
            plt.show()

            # plot Precision-Recall Curve
            
            # predict class values
            yhat = clf.predict(Xv)
            lr_precision, lr_recall, _ = precision_recall_curve(yv, lr_probs)
            lr_f1, lr_auc = f1_score(yv, yhat), auc(lr_recall, lr_precision)
            # summarize scores
            print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
            # plot the precision-recall curves
            no_skill = len(yv[yv==1]) / len(yv)
            plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
            plt.plot(lr_recall, lr_precision, marker='.', label= str(clf).replace('()',''), color='#C10043')
            # axis labels
            plt.xlabel('Recall')
            plt.xlim([0, 1])
            plt.ylabel('Precision')
            plt.ylim([0, 1])
            # show the legend
            plt.legend()
            # add a title
            plt.title('Precision-Recall Curve for ' + str(clf).replace('()',''))
            # show the plot
            plt.show()

            # plot confusion matrix
            plotconfmatrix(yv, yp, clf)
            
            
    avgerr = sum(errs) / len(errs)
    avgacc = sum(acc) / len(acc)
    
    return avgerr, avgacc


def training(Xt, yt, Xv, yv, clf, _plot = False, oversmpl = False):


    if oversmpl:
        Xt, yt = oversample(Xt, yt)
        Xv, yv = oversample(Xv, yv)

    clf.fit(Xt, yt)

    yp = clf.predict(Xv)
    
    acc = accuracy_score(yv, yp)
    
    wrong = np.ones_like(yp)
    wrong[yp == yv] = 0
    err = sum(wrong) / len(yv)
    
    if _plot:
        print(f'train (0, 1): ({len(yt[yt==0])}, {len(yt[yt==1])})')
        print(f'test (0, 1): ({len(yv[yv==0])}, {len(yv[yv==1])})')
        # plot ROC curve
        ns_probs = [0 for _ in range(len(yv))]
        lr_probs = clf.predict_proba(Xv)
        lr_probs = lr_probs[:, 1]
        ns_auc = roc_auc_score(yv, ns_probs)
        lr_auc = roc_auc_score(yv, lr_probs)
        # summarize scores
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Logistic: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(yv, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(yv, lr_probs)
        # plot the roc curve for the model
        mpl.style.use('default')
        plt.figure()
        plt.plot(ns_fpr, ns_tpr, linestyle='--', color='#00325F', label='No Skill')
        plt.plot(lr_fpr, lr_tpr,  label= str(clf).split('(')[0] + ' (area = %0.2f)' % lr_auc, color='#C10043')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend(loc="lower right")
        plt.title('ROC Curve for ' + str(clf).split('(')[0])

        # show the plot
        plt.show()

        # plot Precision-Recall Curve
        
        # predict class values
        yhat = clf.predict(Xv)
        lr_precision, lr_recall, _ = precision_recall_curve(yv, lr_probs)
        lr_f1, lr_auc = f1_score(yv, yhat), auc(lr_recall, lr_precision)
        # summarize scores
        print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
        # plot the precision-recall curves
        no_skill = len(yv[yv==1]) / len(yv)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(lr_recall, lr_precision, marker='.', label= str(clf).split('(')[0], color='#C10043')
        # axis labels
        plt.xlabel('Recall')
        plt.xlim([0, 1])
        plt.ylabel('Precision')
        plt.ylim([0, 1])
        # show the legend
        plt.legend()
        # add a title
        plt.title('Precision-Recall Curve for ' + str(clf).split('(')[0])
        # show the plot
        plt.show()

        # plot confusion matrix
        plotconfmatrix(yv, yp, clf)
    
    return err, acc

def train_and_print(X, y, clf, featureset, _plot = False, oversmpl = True):
    avgerr, avgacc = k_fold(X, y, clf, _plot = _plot, oversmpl = oversmpl)
    print(f'{str(clf).replace("()","")} classifier on {featureset}:')
    print(f'average cross-validation error = {avgerr}, average cross-validation accuracy = {avgacc}', end='\n\n')


# refX features
X_refx = refx_features()

# load essentialities
e = essentialities()

ts = [0, 0.5]


# flow profile features
X_flow = flow_profiles_training_data()


for t in ts:
    y = binary_labels(e, t).astype('int')
    
    # refX predictions
    X_train, X_test, y_train, y_test = train_test_split(X_refx, y, test_size=0.5, random_state=842)
    
    # # flow profile predictions
    # X_train, X_test, y_train, y_test = train_test_split(X_flow, y, test_size=0.5, random_state=842)

    print(f'\nthreshold {t}')
    # support vector machine:
    svm_clf = svm.SVC(probability=True) # svm.SVC(decision_function_shape='ovo')
    err, acc = training(X_train, y_train, X_test, y_test, svm_clf, _plot=True, oversmpl=True)
    print(f'SVM: Error is {err}, Accuracy is {acc}')
    
    # logistic regression:
    logreg = LogisticRegression(random_state=0)
    # train_and_print(X_train, y_train, logreg, 'refx features', _plot = True, oversmpl = True)
    err, acc = training(X_train, y_train, X_test, y_test, logreg, _plot=True, oversmpl=True)
    print(f'LogReg: Error is {err}, Accuracy is {acc}')
    # models = ['BT-549', 'HCT-116','K-562', 'MCF7', 'OVCAR-5']
    

