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


from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
from sklearn.metrics import roc_auc_score

# get true labels for classification
def true_labels():
    fpath = '../data/rolx-train-val-test/'
    yc = np.load(f'{fpath}yctrain.npy')
    return yc

# prediction from rolx
def rolx_training_data():
    fpath = '../data/rolx-train-val-test/'
    X = np.load(f'{fpath}Xtrain.npy')
    y = np.load(f'{fpath}ytrain.npy')
    yc = np.load(f'{fpath}yctrain.npy')
    return X, y, yc

def rolx_in_features():
    models = ['BT-549', 'HCT-116', 'K-562', 'MCF7', 'OVCAR-5']
    
    X = np.load(f'../data/rolx-features/{models[0]}-common-rolx-features-X.npy')
    for model in models[1:]:
        mpath = f'../data/rolx-features/{model}-common-rolx-features-X.npy'
        X = np.vstack([X, np.load(mpath)])
        
    return X

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
    X1 = np.repeat(X[y==1], int(len(X0) / len(X1)), axis=0)
    

    X = np.hstack((X0.T, X1.T)).T

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
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for ' + str(clf).replace('()',''))
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

def train_and_print(X, y, clf, featureset, _plot = False, oversmpl = True):
    avgerr, avgacc = training(X, y, clf, _plot = _plot, oversmpl = oversmpl)
    print(f'{str(clf).replace("()","")} classifier on {featureset}:')
    print(f'average cross-validation error = {avgerr}, average cross-validation accuracy = {avgacc}', end='\n\n')


# rolX predictions
Xr, yr, ycr = rolx_training_data()
ts = [0, 1]

# # flow profiles predictions
Xf, Yf, yf, ycf = flow_profiles_training_data('BT-549')
Xr = Xf
ycr = ycf

for t in ts:
    yb = np.zeros_like(ycr) # binary y: 0 for non-essential (e < 0.5) 1 for essential (e > 0.5)
    yb[ycr > t] = 1

    print(f'\nthreshold {t}')
    # support vector machine:
    svm_clf = svm.SVC(probability=True) # svm.SVC(decision_function_shape='ovo')
    train_and_print(Xr, yb, svm_clf, 'rolx roles', _plot = True, oversmpl = True)
    
    # logistic regression:
    logreg = LogisticRegression(random_state=0)
    train_and_print(Xr, yb, logreg, 'rolx roles', _plot = True, oversmpl = True)
    
    # models = ['BT-549', 'HCT-116','K-562', 'MCF7', 'OVCAR-5']
    

