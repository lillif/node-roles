import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import seaborn as sns
import sys

K = 10

X = np.load('data/train-test-set-rolX/Xtrain.npy')
y = np.load('data/train-test-set-rolX/ytrain.npy')
yc = np.load('data/train-test-set-rolX/yctrain.npy')

X0 = X[yc==0]
X1 = X[yc==1]
X2 = X[yc==2]
X3 = X[yc==3]

X1 = np.repeat(X1, int(len(X0) / len(X1)), axis=0)
X2 = np.repeat(X2, int(len(X0) / len(X2)), axis=0)
X3 = np.repeat(X3, int(len(X0) / len(X3)), axis=0)
X3 = np.hstack((X3.T, X3[0:280].T)).T

y0 = np.repeat(0,len(X0))
y1 = np.repeat(1,len(X1))
y2 = np.repeat(2,len(X2))
y3 = np.repeat(3,len(X3))

X = np.hstack((X0.T, X1.T, X2.T, X3.T)).T
yc = np.hstack((y0,y1,y2,y3))

p = np.random.RandomState(seed=847).permutation(len(yc))
Xsh = X[p]
# ysh = y[p]
ycsh = yc[p]

print(len(Xsh))
print(len(ycsh))

splits = [int(i * len(y) / 10) for i in range(11)]

errors = {}
avgerrors = {}

accs = {}
avgaccs = {}

# for d in range(1,25):
for d in [8]:
    errs = []
    acc = []
    m = True
    for k in range(K):
        
        forest = RandomForestClassifier(max_depth=d, random_state=0)
        
        Xt = np.concatenate((Xsh[:splits[k]], Xsh[splits[k+1]:]))
        yt = np.concatenate((ycsh[:splits[k]], ycsh[splits[k+1]:]))
        
        Xv = Xsh[splits[k]:splits[k+1]]
        yv = ycsh[splits[k]:splits[k+1]]
        
        forest.fit(Xt, yt)
    
        yp = forest.predict(Xv)
        
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
            plt.title('Confusion Matrix for Random Forest Model')
            plt.show()

        
        
    
    errors[d] = errs
    avgerrors[d] = sum(errs) / len(errs)
    
    accs[d] = acc
    avgaccs[d] = sum(acc) / len(acc)

