import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# models = ['BT-549', 'HCT-116','K-562', 'MCF7', 'OVCAR-5']

# for model in models:
#     folder = 'data/' + model + '/'
#     fnames = folder + model
    
#     X = np.load(fnames + '-X.npy')
    
#     n = pd.read_csv(fnames + '_nodes.csv')
#     y = np.array(list(n['essentiality'])) 
#     y[y<1e-14] = 0
    
#     np.save(fnames + '-y-num.npy', y)
    
#     y_classes = y.copy()
#     y_classes[y_classes==1] = -4
#     y_classes[y_classes>0.5] = -3
#     y_classes[y_classes>0.1] = -2
#     y_classes[y_classes>-0.1] = -1
#     y_classes = - y_classes - 1
    
#     np.save(fnames + '-y-classes.npy', y_classes)

models = ['BT-549', 'HCT-116','K-562', 'MCF7', 'OVCAR-5']
ys = {}
Xs = {}
ycs = {}
allycs = np.array([])
allys = np.array([])
for model in models:
    fpath = 'data/training-data-from-rolX/'
    ys[model] = np.load(fpath + model + '-y-num.npy')
    Xs[model] = np.load(fpath + '6roles/' + model + '-6roles-X.npy')
    ycs[model] = np.load(fpath + model + '-y-classes.npy')
    allycs = np.append(allycs, ycs[model])
    allys = np.append(allys, ys[model])

allXs = np.concatenate((Xs['BT-549'], Xs['HCT-116'], Xs['K-562'], Xs['MCF7'], Xs['OVCAR-5']))
seeds = [224, 347, 482, 592]

classes = {}
ps = {}
tr = {}
te = {}

ttr = []
tte = []

for i, s in zip(range(4), seeds):
    classes[i] = np.where(allycs == i)[0]
    ps[i] = np.random.RandomState(seed=s).permutation(np.where(allycs == i)[0])
    split = int(0.8*len(classes[i]))
    tr[i] = sorted(ps[i][:split])
    te[i] = sorted(ps[i][split:])
    ttr += sorted(ps[i][:split])
    tte += sorted(ps[i][split:])
    
ttr = sorted(ttr)
tte = sorted(tte)


Xtrain = allXs[ttr]
yctrain = allycs[ttr]
ytrain = allys[ttr]

Xtest = allXs[tte]
yctest = allycs[tte]
ytest = allys[tte]

# Split features and target into train and test sets
# need the manual version so we get ycs and ys alongside each other
# X_train, X_test, y_train, y_test = train_test_split(allXs, allycs, random_state=1412, stratify=allycs)



fpath = 'data/train-test-set-rolX/'
np.save(fpath + 'Xtrain.npy', Xtrain)
np.save(fpath + 'ytrain.npy', ytrain)
np.save(fpath + 'yctrain.npy', yctrain)
np.save(fpath + 'Xtest.npy', Xtest)
np.save(fpath + 'ytest.npy', ytest)
np.save(fpath + 'yctest.npy', yctest)


