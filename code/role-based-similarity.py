import numpy as np
import networkx as nx
import pickle
import pandas as pd
from numpy.linalg import (norm,
                          matrix_power)

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import community as community_louvain
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

### following "Role-based similarity in directed networks" by Kathryn Cooper and Mauricio Barahona


def directed_diameter(e, GG):    
    # largest strongly connected component:
    lscc = max(nx.strongly_connected_components(GG), key=len)
    Gr = nx.DiGraph()
    Gr.add_edges_from([s,t] for s,t in zip(e['Source'], e['Target']) 
                      if (s in lscc and t in lscc))
    k_max = nx.algorithms.distance_measures.diameter(Gr)
    return k_max

def flow_profiles(A, k_max=5, alpha=0.5):
    
    N = A.shape[0] # number of nodes
    
    X = np.zeros([N, 2*k_max])
    
    ones = np.ones([N,1])
    
    eigval = max(np.linalg.eig(A)[0]).real # largest eigenvalue of A
    beta = alpha / eigval
    beta_A = beta * A

    for k in range(k_max):
        # incoming - use k+1 as range is from 0 to 10, we need 1 to 11
        
        X[:,0+k] = (matrix_power(beta_A.T, k+1)@ones).squeeze()
        # outgoing
        X[:,k_max+k] = (matrix_power(beta_A, k+1)@ones).squeeze()

    Y = cosine_similarity(X)
    
    return X, Y


def edges_to_denseMFG(filename):

    # weighted:
    e = pd.read_csv(filename)
    n = pd.read_csv(filename.replace('edges', 'nodes'))
    N = len(n)
    
    A = np.zeros([N,N])
    l = list(n['id'])
    
    G = nx.DiGraph()
    
    for row in e.iterrows():
        _, s, t, w = row[1]
        G.add_edge(s,t, weight=w)
        s = l.index(s)
        t = l.index(t)
        A[s, t] = w
    
    k_max = directed_diameter(e, G) # diameter of largest strongly connected component of MFG
    
    return A, k_max

def draw_graph(Y):
    GY = nx.convert_matrix.from_numpy_array(Y)
    partition = community_louvain.best_partition(GY)
    
    
    plt.figure(figsize=[13, 10])
    # draw the graph
    # pos = nx.spring_layout(GY)
    pos=nx.kamada_kawai_layout(GY)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(GY, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(GY, pos, alpha=0.5, width=0.02)
    plt.show()


models = ['BT-549', 'HCT-116','K-562', 'MCF7', 'OVCAR-5']

for model in models:
    
    A, k_max = edges_to_denseMFG(f'../data/mfgs/{model}_edges.csv')
    print(k_max)
    # A = pickle.load( open( "denseMFG.p", "rb" ) )
    # k_max = 11
    X, Y = flow_profiles(A, k_max, alpha=0.8)
    
    fpath = '../data/role-based-sim-features/'
    
    np.save(f'{fpath}{model}-X.npy', X)
    np.save(f'{fpath}{model}-Y.npy', Y)
    # clustering = SpectralClustering(assign_labels="discretize").fit(Y)
    
    # specify n_clusters = 4; otherwise default is 8
# clustering = SpectralClustering(n_clusters=4, assign_labels="discretize").fit(Y)
# clusters = clustering.labels_

# sc = SpectralClustering(n_clusters=4, affinity='precomputed', n_init=100,
#                         assign_labels='discretize')
# sc.fit_predict(Y) # 

# draw_graph(Y)
