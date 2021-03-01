import numpy as np
import matplotlib.pyplot as plt
import pickle

# importing models from sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# sns for the confusion matrix plot
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# importing sys for exiting when necessary
import sys

# prediction from rolx
def rolx_training_data():
    X = np.load('../data/rolx-train-val-test/Xtrain.npy')
    y = np.load('../data/rolx-train-val-test/ytrain.npy')
    yc = np.load('../data/rolx-train-val-test/yctrain.npy')
    return X, y, yc

# prediction from flow profiles
def flow_profiles_training_data():
    X = pickle.load( open( "data/BT-549/role-based-sim/X.p", "rb" ) )
    Y = pickle.load( open( "data/BT-549/role-based-sim/Y.p", "rb" ) )
    y = np.load('data/BT-549/y_num_RolX.npy')
    yc = np.load('data/BT-549/y_classes_RolX.npy')
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


def training(X, y, clf, K = 5, m = False, oversmpl = False):

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
        
        if m:
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
            plt.title('Confusion Matrix for ' + str(clf).replace('()',''))
            plt.show()
            

    avgerr = sum(errs) / len(errs)
    avgacc = sum(acc) / len(acc)
    
    return avgerr, avgacc



# classifiers
classifiers = [svm.SVC(), RandomForestClassifier(), MLPClassifier()]

# # rolX predictions
Xr, yr, ycr = rolx_training_data()

for clf in classifiers:
    training(Xr, ycr, clf, m = True, oversmpl = True)

# training(Xo, yco, svm.SVC(decision_function_shape='ovo'))
# training(Xo, yco, svm.SVC())


# flow profiles redictions
# Xf, Yf, yf, ycf = flow_profiles_training_data()

# Xo, yco = oversample(Xf, ycf)
# Xu, ycu = undersample(Xf, ycf, ncps = 3)




