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

# might look cool
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

# manually get the diameter of the directed graph
def directed_diameter(e, GG):    
    # largest strongly connected component:
    lscc = max(nx.strongly_connected_components(GG), key=len)
    Gr = nx.DiGraph()
    Gr.add_edges_from([s,t] for s,t in zip(e['Source'], e['Target']) 
                      if (s in lscc and t in lscc))
    k = nx.algorithms.distance_measures.diameter(Gr)
    return k

# compute the similarity
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


def saveXy(X, y, alpha, fpath):

    fpath = '../data/role-based-sim-features/'

    np.save('{0}{1}-alpha{2:.1f}-X.npy'.format(fpath, model, alpha), X)
    np.save('{0}{1}-alpha{2:.1f}-Y.npy'.format(fpath, model, alpha), Y)


def compare_cluster_to_essentialty(clusters, e_groups):
    """
    clusters: 1 x N array of clusters which each node was assigned to
    e_groups: 1 x N array of essentiality groups each node belongs to
    """

    # assign clusters to most frequent e_group:
    ce = {}
    for c in set(clusters):
        print(np.bincount(e_groups[clusters == c]))
        ce[c] = np.argmax(np.bincount(e_groups[clusters == c]))
    print('for alpha {0:.1f} the c-e assignments are {1}'.format(alpha, ce))

# models = ['BT-549', 'HCT-116','K-562', 'MCF7', 'OVCAR-5']

# start for BT-549
model = 'BT-549'

# try alphas in the whole possible range
for alpha in np.arange(0.1, 1.1, 0.1):
# for alpha in [0.8]:

    A, k = edges_to_denseMFG(f'../data/mfgs/{model}_edges.csv')
    X, Y = sim_rb(A, k, alpha)

    # clustering = SpectralClustering(assign_labels="discretize").fit(Y)

    # # specify n_clusters = 4; otherwise default is 8
    clustering = SpectralClustering(n_clusters=4, assign_labels="discretize").fit(Y)
    clusters = clustering.labels_

    sc = SpectralClustering(n_clusters=4, affinity='precomputed', n_init=100,
                            assign_labels='discretize')
    sc.fit_predict(Y) # 
    clusters = sc.labels_

    e_groups = np.load(f'../data/rolx-memberships/{model}-y-classes.npy')
    e_groups = e_groups.astype('int64')
    
    compare_cluster_to_essentialty(clusters, e_groups)
    # draw_graph(Y)
