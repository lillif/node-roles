import numpy as np
import networkx as nx
import pickle
import pandas as pd
from numpy.linalg import norm

from sklearn.cluster import SpectralClustering

def directed_diameter(e, GG):    
    # largest strongly connected component:
    lscc = max(nx.strongly_connected_components(GG), key=len)
    Gr = nx.DiGraph()
    Gr.add_edges_from([s,t] for s,t in zip(e['Source'], e['Target']) if (s in lscc and t in lscc))
    k = nx.algorithms.distance_measures.diameter(Gr)
    return k

def sim_rb(A, k=5, alpha=0.5):

    N = A.shape[0] # number of nodes
    
    X = np.zeros([N, 2*k])
    
    ones = np.ones([N,1])
    
    lambda1 = max(np.linalg.eig(A)[0]).real # largest eigenvalue of A
    alpha = 1 # for now: equal weight for local and global environment
    beta = alpha / lambda1
    sA = beta * A
    
    for kk in range(k):
        # incoming
        X[:,0+kk] = (sA.T@ones).T
        # outgoing
        X[:,k+kk] = (sA@ones).T
    
    # prevent division by 0 (no 0 vectors!) by using dense matrix
    Y = np.zeros(A.shape)
    for i in range(len(A)):
        for j in range(len(A)):
            Y[i,j] = X[i,:]@X[j,:].T / (norm(X[i,:]) * norm(X[j,:]))
    return X, Y



# load MFG

def edges_to_denseMFG(filename):
    filename = 'BT-549edges.csv'
    # weighted:
    e = pd.read_csv(filename)
    GG = nx.DiGraph()
    
    for row in e.iterrows():
        s, t, w = row[1]
        GG.add_edge(s,t, weight=w)
    
    # k = directed_diameter(e, GG) # already computed: kmax = 11
    k = 11 # diameter of largest strongly connected component of MFG
    
    A = nx.linalg.graphmatrix.adjacency_matrix(GG).toarray()
    return A, k

# A, k = edges_to_denseMFG('BT-549edges.csv')
A = pickle.load( open( "denseMFG.p", "rb" ) )
k = 11
X, Y = sim_rb(A, k)


clustering = SpectralClustering(n_clusters=4, assign_labels="discretize").fit(Y)
clusters = clustering.labels_
