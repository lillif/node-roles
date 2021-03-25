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


# sklearn preprocessing functions
from sklearn.model_selection import (train_test_split, StratifiedKFold)
from sklearn.preprocessing import (StandardScaler, MinMaxScaler)

# analysing and visualising classifiier predictions
import seaborn as sns
from sklearn.metrics import (accuracy_score,
                             auc,
                             classification_report,
                             confusion_matrix,
                             f1_score,
                             plot_roc_curve,
                             precision_recall_curve, 
                             roc_auc_score,
                             roc_curve)

import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt




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
    
def k_fold_roc(X, y, cv, classifier, featureset, ax):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    

    
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(classifier, X[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=f'ROC {featureset} {str(clf).split("(")[0]}')
    ax.legend(loc="lower right", fontsize='small')



def k_fold_pr(X, y, cv, clf, featureset, ax):
    y_real = []
    y_proba = []

    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        clf.fit(Xtrain, ytrain)
        pred_proba = clf.predict_proba(Xtest)
        precision, recall, _ = precision_recall_curve(ytest, pred_proba[:,1])
        lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
        ax.plot(recall, precision, label=lab, lw=0.5)
        y_real.append(ytest)
        y_proba.append(pred_proba[:,1])
        
    no_skill = len(y[y==1]) / len(y)
    ax.plot([0, 1], [no_skill, no_skill], linestyle='--', lw=2, color='r', label='No Skill')
    
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'Overall AUC=%.4f' % (auc(recall, precision))
    ax.plot(recall, precision, label=lab, lw=2, color='b')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left', fontsize='small')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=f'Precision Recall {featureset} {str(clf).split("(")[0]}')


def k_fold_roc1(X, y, classifier, ax, i, mean_fpr):
    viz = plot_roc_curve(classifier, X, y,
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0

    return interp_tpr, viz.roc_auc

   
def k_fold_pr1(X, y, clf, ax, i):
    pred_proba = clf.predict_proba(X)
    precision, recall, _ = precision_recall_curve(y, pred_proba[:,1])
    lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
    ax.plot(recall, precision, label=lab, lw=0.5)
    return y, pred_proba[:,1]

        
def k_fold_training(X, y, cv, clf, featureset):
    mpl.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    #ROC
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    # PR
    y_real = []
    y_proba = []
    
    for i, (train, test) in enumerate(cv.split(X, y)):
        clf.fit(X[train], y[train])
        t, a = k_fold_roc1(X[test], y[test], clf, axes[0], i, mean_fpr)
        tprs.append(t)
        aucs.append(a)
        
        yr, yp = k_fold_pr1(X[test], y[test], clf, axes[1], i)
        y_real.append(yr)
        y_proba.append(yp)
        
    # FINAL ROC    
    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
            title=f'ROC {featureset} {str(clf).split("(")[0]}')
    ax.legend(loc="lower right", fontsize='small')

    # FINAL PR
    ax = axes[1]
    no_skill = len(y[y==1]) / len(y)
    ax.plot([0, 1], [no_skill, no_skill], linestyle='--', lw=2, color='r', label='No Skill')
    
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'Overall AUC=%.4f' % (auc(recall, precision))
    ax.plot(recall, precision, label=lab, lw=2, color='b')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left', fontsize='small')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=f'Precision Recall {featureset} {str(clf).split("(")[0]}')

    fig.tight_layout()
    plt.show()
        
        
# essentiality labels
e = essentialities()
t = 0.1
y = binary_labels(e, t).astype('int')

# reaction features (uncomment the one to use)
# featureset = 'ReFeX Features'
# X = refx_features()

featureset = 'Flow Profiles'
X = flow_profiles_training_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=842, stratify=y)

# optional: standardise / normalise features
# featureset += ' (normalised)'
# normalize = StandardScaler().fit(X_train)
# X_train = normalize.transform(X_train)
# X_test = normalize.transform(X_test)

# featureset += ' (standardised)'
# standardize = MinMaxScaler()
# X_train = standardize.fit_transform(X_train)
# X_test = standardize.transform(X_test)


classifiers = [svm.SVC(probability=True),
                LogisticRegression(random_state=0),
                MLPClassifier(random_state=0),
                RandomForestClassifier()]

cv = StratifiedKFold(n_splits=5)
for clf in [classifiers[3]]:
    # err, acc = training(X_train, y_train, X_test, y_test, clf, featureset, _plot=True, oversmpl=False)
    k_fold_training(X_train, y_train, cv, clf, featureset)
    # print(f'{str(clf).split("(")[0]}: Error is {err}, Accuracy is {acc}')
    
# y = quaternary_labels(e).astype('int')
# X_train, X_test, y_train, y_test = train_test_split(X_refx, y, test_size=0.2, random_state=842, stratify=y)
# refx_scaler = StandardScaler().fit(X_train)
# X_train = refx_scaler.transform(X_train)
# X_test = refx_scaler.transform(X_test)

# for clf in classifiers:
#     err, acc = training(X_train, y_train, X_test, y_test, clf, featureset, _plot=True, oversmpl=False, four=True)



