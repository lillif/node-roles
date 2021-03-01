import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.neural_network import MLPClassifier

import seaborn as sns

K = 10

X = np.load('data/train-test-set-rolX/Xtrain.npy')
y = np.load('data/train-test-set-rolX/ytrain.npy')
yc = np.load('data/train-test-set-rolX/yctrain.npy')

p = np.random.RandomState(seed=847).permutation(len(y))
Xsh = X[p]
ysh = y[p]
ycsh = yc[p]

splits = [int(i * len(y) / 10) for i in range(11)]

errors = {}
avgerrors = {}

accs = {}
avgaccs = {}

# for d in range(1,25):
errs = []
acc = []
for k in range(K):

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    
    Xt = np.concatenate((Xsh[:splits[k]], Xsh[splits[k+1]:]))
    yt = np.concatenate((ycsh[:splits[k]], ycsh[splits[k+1]:]))
    
    Xv = Xsh[splits[k]:splits[k+1]]
    yv = ycsh[splits[k]:splits[k+1]]
    
    clf.fit(Xt, yt)
    
    yp = clf.predict(Xv)

    acc.append(accuracy_score(yv, yp))
    
    wrong = np.ones_like(yp)
    wrong[yp == yv] = 0
    errs.append(sum(wrong) / len(yv))
    
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
    plt.title('Confusion Matrix for Logistic Regression Model')
    plt.show()
        
# avgerr = sum(errs) / len(errs)
# avgacc = sum(acc) / len(acc)


