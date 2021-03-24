import pandas as pd
import networkx as nx

import numpy as np

### following dkaslovsky's example from
### https://github.com/dkaslovsky/GraphRole/blob/master/examples/example.ipynb


def features(model):
    mfeatures = pd.read_csv(f'../data/rolx-features/{model}_features-df.csv')
    return mfeatures
    
def commonfeatures(dfs):
    cols = [set(d.columns) for d in dfs.values()]
    inall = set.intersection(*cols) # need * for list of sets in intersection
    for k, v in dfs.items(): dfs[k] = v[inall]
    return dfs, cols, inall

models = ['BT-549', 'HCT-116', 'K-562', 'MCF7', 'OVCAR-5']
dfs = {}
for model in models:
    dfs[model] = features(model)

feat, cols, inall = commonfeatures(dfs)


for model in models:
    X_features = dfs[model].to_numpy()
    np.save(f'../data/rolx-features/{model}-common-rolx-features-X.npy', X_features)
