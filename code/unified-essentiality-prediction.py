import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

# sklearn classification algorithms
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


# sklearn regression algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm

# sklearn preprocessing functions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, MinMaxScaler)

# analysing and visualising classifiier predictions
import seaborn as sns
from sklearn.metrics import (accuracy_score,
                             auc,
                             confusion_matrix,
                             f1_score,
                             precision_recall_curve, 
                             roc_auc_score,
                             roc_curve)
# from sklearn.metrics import classification_report

import pandas as pd

from sklearn.utils import shuffle




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

def tertiary_labels(e):
    """
    Parameters
    ----------
    e : np.array (1-dimensional)
        essentiality values.

    Returns
    -------
    c : np.array (1-dimensional)
        three essentiality classes.

    """

    c = np.zeros_like(e)
    c[(e >= 0.1) & (e < 1)] = 1
    c[e == 1] = 2
    return c
def quaternary_labels(e):
    c = np.zeros_like(e)
    c[(e >= 0.1) & (e < 0.5)] = 1
    c[(e >= 0.5) & (e < 1)] = 2
    c[e == 1] = 3
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

def plotconfmatrix4(yv, yp, clf):
    # plot confusion matrix
    # Get and reshape confusion matrix data
    matrix = confusion_matrix(yv, yp)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    
    # Build the plot
    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Blues, linewidths=0.2)
    
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


    
def plotroc(Xv, yv, clf, featureset):
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
    plt.title(f"ROC Curve for {str(clf).split('(')[0]} on {featureset} features")

    # show the plot
    plt.show()

def plotpr(Xv, yv, clf, featureset):
    lr_probs = clf.predict_proba(Xv)
    lr_probs = lr_probs[:, 1]
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
    plt.title(f"Precision-Recall Curve for {str(clf).split('(')[0]} on {featureset} features")
    # show the plot
    plt.show()



def training(Xt, yt, Xv, yv, clf, featureset, _plot = False, oversmpl = False, four=False):

    if oversmpl:
        Xt, yt = oversample(Xt, yt)
        Xv, yv = oversample(Xv, yv)

    clf.fit(Xt, yt)

    yp = clf.predict(Xv)
    
    acc = accuracy_score(yv, yp)
    
    wrong = np.ones_like(yp)
    wrong[yp == yv] = 0
    err = sum(wrong) / len(yv)
    
    if _plot and not four:
        print(f'train (0, 1): ({len(yt[yt==0])}, {len(yt[yt==1])})')
        print(f'test (0, 1): ({len(yv[yv==0])}, {len(yv[yv==1])})')
        # plot ROC curve
        plotroc(Xv, yv, clf, featureset)
        
        # plot Precision-Recall Curve
        plotpr(Xv, yv, clf, featureset)
        
        # plot confusion matrix
        plotconfmatrix(yv, yp, clf)
        
    elif _plot:
        plotconfmatrix4(yv, yp, clf)
    
    return err, acc

def visualise_classes(y, e):
    plt.figure()
    nz = np.count_nonzero(y)
    plt.bar([0,1], [len(y) - nz, nz], width=0.2)
    plt.xticks([0,1], labels=['Non-Essential', 'Essential'])
    plt.xlabel('Essentiality Class')
    plt.ylabel('Number of Reactions')
    plt.title('Number of Reactions per Class (binary)')
    
    plt.figure()
    nc = len(e[e < 0.1])
    mc = len(e[(e >= 0.1) & (e < 0.5)])
    sc = len(e[(e >= 0.5) & (e < 1)])
    le = len(e[e == 1])
    plt.bar([0,1, 2, 3], [nc, mc, sc, le], width=0.2)
    plt.xticks([0,1, 2, 3], labels=['No Change', 'Mild Change', 'Severe Change', 'Lethal'])
    plt.xlabel('Essentiality Class')
    plt.ylabel('Number of Reactions')
    plt.title('Number of Reactions per Class (multiclass)')

    

def visualise_data(e, X, dataset='RefX'):
    mpl.style.use('default')
    plt.figure()
    yy = sorted(e)
    xx = np.arange(len(yy))
    plt.plot(xx,yy,'x-', color='#00325F')
    # plt.plot(xx,yy,'x-', color='#C10043')
    #C10043
    plt.xlabel('Reaction')
    plt.title('Reaction Essentialities')
    plt.ylabel('Essentiality')
    plt.show()
    
    # boxplot
    plt.figure()
    data = [X[:,i] for i in range(X.shape[1])]
    plt.boxplot(data)
    plt.xlabel('Feature Dimension')
    plt.ylabel('Values')
    plt.title(f'{dataset} Feature Values by Dimension')
    plt.show()
    
    # visualise correlation matrix of dimensions
    plt.figure()
    corr = np.corrcoef(X.T)
    sns.heatmap(corr)
    plt.title(f'{dataset} Correlation of Feature Dimensions')
    plt.xlabel('Dimension')
    plt.ylabel('Dimension')
    plt.show()
    

# refX features
X_refx = refx_features()


# load essentialities
e = essentialities()

# ts = [0, 0.5]
ts = [0.5]

# flow profile features
X_flow = flow_profiles_training_data()

classifiers = [svm.SVC(probability=True),
                LogisticRegression(random_state=0),
                MLPClassifier(random_state=0),
                RandomForestClassifier()]

t = 0.1

y = binary_labels(e, t).astype('int')

# # refX predictions
# featureset = 'raw RefX'
# X_train, X_test, y_train, y_test = train_test_split(X_refx, y, test_size=0.2, random_state=842, stratify=y)

# # standardized refX predictions
# featureset = 'std RefX'
# X_train, X_test, y_train, y_test = train_test_split(X_refx, y, test_size=0.2, random_state=842, stratify=y)
# refx_scaler = StandardScaler().fit(X_train)
# X_train = refx_scaler.transform(X_train)
# X_test = refx_scaler.transform(X_test)

# flow profile predictions
featureset = 'Normalised Flow Profile'
X_train, X_test, y_train, y_test = train_test_split(X_flow, y, test_size=0.2, random_state=842, stratify=y)
flow_scaler = StandardScaler().fit(X_train)
X_train = flow_scaler.transform(X_train)
X_test = flow_scaler.transform(X_test)
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

for clf in classifiers:
    err, acc = training(X_train, y_train, X_test, y_test, clf, featureset, _plot=True, oversmpl=False)
    print(f'{str(clf).split("(")[0]}: Error is {err}, Accuracy is {acc}')
    
# y = quaternary_labels(e).astype('int')
# X_train, X_test, y_train, y_test = train_test_split(X_refx, y, test_size=0.2, random_state=842, stratify=y)
# refx_scaler = StandardScaler().fit(X_train)
# X_train = refx_scaler.transform(X_train)
# X_test = refx_scaler.transform(X_test)

# for clf in classifiers:
#     err, acc = training(X_train, y_train, X_test, y_test, clf, featureset, _plot=True, oversmpl=False, four=True)
    



# rf = RandomForestRegressor(random_state = 42)

# Xs, es = shuffle(X_refx, e,  random_state=0)
# split = int(0.8*len(Xs))
# X_train = Xs[:split]
# X_test = Xs[split:]
# y_train = es[:split]
# y_test = es[split:]

# refx_scaler = StandardScaler().fit(X_train)
# X_train = refx_scaler.transform(X_train)
# X_test = refx_scaler.transform(X_test)


# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)


# ind = np.argsort(y_test)
# sorted_y_test = y_test[ind]
# sorted_y_pred = y_pred[ind]

# xx = np.arange(len(y_test))
# mpl.style.use('default')
# plt.figure()
# plt.plot(xx, sorted_y_pred, label='Predicted Values', color='#00325F')
# plt.plot(xx, sorted_y_test, label='True Values', color='#C10043')
# plt.legend()
# plt.xlabel('Reactions (sorted by ascending true label)')
# plt.ylabel('Essentiality')
# plt.title('Random Forest')

# svc = svm.SVC(kernel='poly', degree=3)
# svc.fit(X_train, y_train)

# y_pred = rf.predict(X_test)


# ind = np.argsort(y_test)
# sorted_y_test = y_test[ind]
# sorted_y_pred = y_pred[ind]

# xx = np.arange(len(y_test))
# mpl.style.use('default')
# plt.figure()
# plt.plot(xx, sorted_y_pred, label='Predicted Values', color='#00325F')
# plt.plot(xx, sorted_y_test, label='True Values', color='#C10043')
# plt.legend()
# plt.xlabel('Reactions (sorted by ascending true label)')
# plt.ylabel('Essentiality')
# plt.title('SVM')
