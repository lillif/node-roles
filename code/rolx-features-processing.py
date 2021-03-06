import pandas as pd
import networkx as nx

import numpy as np

### following dkaslovsky's example from
### https://github.com/dkaslovsky/GraphRole/blob/master/examples/example.ipynb


def features(model):

    mfeatures = pd.read_csv(f'../data/rolx-features/{model}_features-df.csv')
    return mfeatures
    
def commonfeatures(dfs):
    
    cols = {}

    for m, d in dfs.items():
        cols[m] = []
        for c in d.columns:
            cols[m].append(c)
            
    all_ = []
    
    for cc in cols.values():
        for c in cc:
            all_.append(c)
            
    all_ = list(set(all_))
    
    inall = []
    
    for a in all_:
        if a in cols['BT-549'] and a in cols['HCT-116'] and a in cols['K-562']:
            if a in cols['MCF7'] and a in cols['OVCAR-5']:
                inall.append(a)
        
    for k, v in dfs.items():
        dfs[k] = v[inall]
    return dfs

models = ['BT-549', 'HCT-116', 'K-562', 'MCF7', 'OVCAR-5']
dfs = {}
for model in models:
    dfs[model] = features(model)

feat = commonfeatures(dfs)


for model in models:
    X_features = dfs[model].to_numpy()
    np.save(f'../data/rolx-features/{model}-common-rolx-features-X.npy', X_features)
