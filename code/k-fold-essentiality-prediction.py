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
    
def k_fold_confmat(yv, yp):
    # plot confusion matrix
    # Get and reshape confusion matrix data
    matrix = confusion_matrix(yv, yp)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    return matrix
    
def k_fold_plot_confmat(matrices, ax):
    matrix = np.mean(matrices, axis=0)
    
    # Build the plot
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Blues, linewidths=0.2, ax = ax)
    
    # Add labels to the plot
    class_names = ['Non-Essential', 'Essential']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticks(tick_marks2)
    ax.set_yticklabels(class_names)#, rotation=0)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set(title='Confusion Matrix')




def k_fold_roc(X, y, classifier, ax, i, mean_fpr):
    viz = plot_roc_curve(classifier, X, y,
                         name=f'Fold {i}',
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0

    return interp_tpr, viz.roc_auc

   
def k_fold_pr(X, y, clf, ax, i):
    pred_proba = clf.predict_proba(X)
    precision, recall, _ = precision_recall_curve(y, pred_proba[:,1])
    lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
    ax.plot(recall, precision, label=lab, lw=0.5)
    return y, pred_proba[:,1]

        
def k_fold_training(X, y, cv, clf, featureset, cstr):
    mpl.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    #ROC
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    # PR
    y_real = []
    y_proba = []
    
    # CONFMAT
    matrices = []
    
    # accuracy
    acc = []
    
    for i, (train, test) in enumerate(cv.split(X, y)):
        clf.fit(X[train], y[train])
        # t, a = k_fold_roc(X[test], y[test], clf, axes[0], i, mean_fpr)
        # tprs.append(t)
        # aucs.append(a)
        
        yr, yp = k_fold_pr(X[test], y[test], clf, axes[1], i)
        y_real.append(yr)
        y_proba.append(yp)
        
        y_pred = clf.predict(X[test])
        matrices.append(k_fold_confmat(y[test], y_pred))
        acc.append(accuracy_score(y[test], clf.predict(X[test])))
        
    # CONFUSION MATRIX
    # ax = axes[0]
    k_fold_plot_confmat(matrices, axes[0])
        
    # # FINAL ROC
    # ax = axes[0]
    # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #         label='Chance', alpha=.8)
    
    # mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    # ax.plot(mean_fpr, mean_tpr, color='b',
    #         label=r'Mean AUC = %0.2f $\pm$ %0.2f' % (mean_auc, std_auc),
    #         lw=2, alpha=.8)
    
    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                 label=r'$\pm$ 1 std. dev.')
    
    # ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
    #         title='ROC')
    # ax.legend(loc="lower right", fontsize='small')

    # # FINAL PR
    ax = axes[1]
    mpl.style.use('default')
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
           title='Precision Recall')
    plt.suptitle(f'{cstr} on {featureset}')
    fig.tight_layout()
    plt.show()
    return acc
        
# essentiality labels
e = essentialities()
t = 0.1
y = binary_labels(e, t).astype('int')

# # reaction features (uncomment the one to use)
featureset = 'ReFeX Features'
X = refx_features()

# featureset = 'Flow Profiles'
# X = flow_profiles_training_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=842, stratify=y)

# optional: standardise / normalise features
featureset += ' (normalised)'
normalize = StandardScaler().fit(X_train)
X_train = normalize.transform(X_train)
X_test = normalize.transform(X_test)

# featureset += ' (standardised)'
# standardize = MinMaxScaler()
# X_train = standardize.fit_transform(X_train)
# X_test = standardize.transform(X_test)


classifiers = [svm.SVC(probability=True),
                LogisticRegression(random_state=0),
                MLPClassifier(random_state=0),
                RandomForestClassifier()]

cstrings = ['Support Vector Machine',
            'Logistic Regression Classifier',
            'Multi-Layer Perceptron',
            'Random Forest Classifier']

acc = {}

cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=62)
for clf, cstr in zip(classifiers, cstrings):
    acc[cstr] = k_fold_training(X_train, y_train, cv, clf, featureset, cstr)

    

    
mean_acc = [np.mean(acc[k]) for k in acc.keys()]
acc_stdev = [np.std(acc[k]) for k in acc.keys()]
plt.figure()
for i in range(len(mean_acc)):
    plt.errorbar(i, mean_acc[i], ls='', yerr=acc_stdev[i], marker='o',
                 label = cstrings[i])
plt.ylim([0.6,0.9])
plt.xticks(np.arange(4), labels=cstrings, rotation=10)
plt.ylabel('Accuracy')
plt.xlabel('Classifier')
plt.title(f'Cross-Validation Accuracy per Classifier on {featureset}')

print(f'Average accuracies on {featureset}\n{[a + " " + str(np.round(b, decimals=3)) for a,b in zip(cstrings, mean_acc)]}')