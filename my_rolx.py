import numpy as np
import networkx as nx
import pickle
import pandas as pd
from numpy.linalg import norm
from sklearn.decomposition import PCA

# node-feature matrix X
X = pickle.load( open( "X.p", "rb" ) )


N,F = X.shape

for r in range(F):
    # create low-rank approximation of rank r
    pca = PCA(n_components=r)
    pca.fit(X)
    G = pca.transform(X)