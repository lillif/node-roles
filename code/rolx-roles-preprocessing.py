import numpy as np
import pandas as pd


models = ['BT-549', 'HCT-116','K-562', 'MCF7', 'OVCAR-5']

# for model in models:
#     folder = '../data/rolx-memberships/'
#     # fnames = folder + model
    
#     X = np.load(f'{folder}{model}-X.npy')
    
#     n = pd.read_csv(f'../data/{model}_nodes.csv')
#     y = np.array(list(n['essentiality'])) 
#     y[y<1e-14] = 0
    
#     np.save(f'{folder}{model}-y-num.npy', y)
    
#     y_classes = y.copy()
#     y_classes[y_classes==1] = -4
#     y_classes[y_classes>0.5] = -3
#     y_classes[y_classes>0.1] = -2
#     y_classes[y_classes>-0.1] = -1
#     y_classes = - y_classes - 1
    
#     np.save(f'{folder}{model}-y-classes.npy', y_classes)

models = ['BT-549', 'HCT-116','K-562', 'MCF7', 'OVCAR-5']
ys = {}
Xs = {}
ycs = {}
allycs = np.array([])
allys = np.array([])
for model in models:
    fpath = '../data/rolx-memberships/'
    ys[model] = np.load(f'{fpath}{model}-y-num.npy')
    Xs[model] = np.load(f'{fpath}{model}-X.npy')
    ycs[model] = np.load(f'{fpath}{model}-y-classes.npy')
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

fpath = '../data/rolx-train-val-test/'
np.save(f'{fpath}Xtrain.npy', Xtrain)
np.save(f'{fpath}ytrain.npy', ytrain)
np.save(f'{fpath}yctrain.npy', yctrain)
np.save(f'{fpath}Xtest.npy', Xtest)
np.save(f'{fpath}ytest.npy', ytest)
np.save(f'{fpath}yctest.npy', yctest)


