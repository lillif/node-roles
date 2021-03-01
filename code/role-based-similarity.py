import numpy as np
import networkx as nx
import pickle
import pandas as pd
from numpy.linalg import norm

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import community as community_louvain
from sklearn.cluster import SpectralClustering

### following "Role-based similarity in directed networks" by Kathryn Cooper and Mauricio Barahona


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

    # weighted:
    e = pd.read_csv(filename)
    G = nx.DiGraph()
    
    for row in e.iterrows():
        _, s, t, w = row[1]
        G.add_edge(s,t, weight=w)
    
    k = directed_diameter(e, G) # diameter of largest strongly connected component of MFG
    
    A = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
    return A, k


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
    
    A, k = edges_to_denseMFG(f'../data/{model}_edges.csv')
    # A = pickle.load( open( "denseMFG.p", "rb" ) )
    # k = 11
    X, Y = sim_rb(A, k, alpha=0.8)
    
    np.save(f'../data/role-based-sim-features/{model}-X.npy', X)
    np.save(f'../data/role-based-sim-features/{model}-Y.npy', Y)
    
    # clustering = SpectralClustering(n_clusters=4, assign_labels="discretize").fit(Y)
    # clusters = clustering.labels_
    
    # draw_graph(Y)
