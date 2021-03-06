import pandas as pd
import networkx as nx

import numpy as np


from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sns


from graphrole import RecursiveFeatureExtractor, RoleExtractor

import argparse
import sys

### following dkaslovsky's example from
### https://github.com/dkaslovsky/GraphRole/blob/master/examples/example.ipynb


def runRolX(model):

    e = pd.read_csv(f'../data/mfgs/{model}_edges.csv')
    G = nx.DiGraph()

    for row in e.iterrows():
        _, s, t, w = row[1]
        G.add_edge(s,t, weight=w)

    # extract features
    feature_extractor = RecursiveFeatureExtractor(G)
    features = feature_extractor.extract_features()
    
    print(f'\nFeatures extracted from {feature_extractor.generation_count} recursive generations:')
    print(features)
    
    X_features = features.to_numpy()
    np.save(f'../data/rolx-features/{model}-raw-rolx-features-X.npy', X_features)
    features.to_csv(f'../data/rolx-features/{model}_features-df.csv', index=False)
    
    return 

    # assign node roles
    role_extractor = RoleExtractor(n_roles=8)
    role_extractor.extract_role_factors(features)
    node_roles = role_extractor.roles

    print('\nNode role assignments:')
    pprint(node_roles)

    print('\nNode role membership by percentage:')
    print(role_extractor.role_percentage.round(2))

    X = role_extractor.role_percentage.round(2).to_numpy()
    np.save(f'../data/rolx-memberships/{model}-8roles-X.npy', X)
    
    return

    # build color palette for plotting
    unique_roles = sorted(set(node_roles.values()))
    color_map = sns.color_palette('Paired', n_colors=len(unique_roles))
    # map roles to colors
    role_colors = {role: color_map[i] for i, role in enumerate(unique_roles)}
    # build list of colors for all nodes in G
    node_colors = [role_colors[node_roles[node]] for node in G.nodes]

    
    # plot graph
    # pos=nx.kamada_kawai_layout(G)
    # # plot graph
    # plt.figure(figsize=[13, 10])
    # nx.draw(G, pos=pos, with_labels=False, 
    #         node_color=node_colors, node_size=80, edge_color="grey", width = 0.5)
    # plt.show()

    # export as csv for gephi

    n = pd.read_csv(f'../data/mfgs/{model}_nodes.csv')
    nr_df = list(node_roles.values())
    n['Role'] = nr_df
    n.to_csv(f'../data/mfgs/{model}_rolxnodes.csv', index=False)


    # plotting roles & essentiality
    essentialities = {}
    for i, e in zip(n['id'], n['essentiality']):
        essentialities[i] = e
        if e < 1e-14:
            essentialities[i] = 0.0
        

    role_ess = {}
    for role in unique_roles:
        role_ess[role] = [0,0,0,0]

    for role, e in essentialities.items():
        if e == 1.0:
            role_ess[node_roles[role]][3] += 1
        elif e > 0.5:
            role_ess[node_roles[role]][2] += 1
        elif e > 0.1:
            role_ess[node_roles[role]][1] += 1
        else:
            role_ess[node_roles[role]][0] += 1
        

    plt.figure()
    r = len(unique_roles) # number of roles 
    ind = np.arange(r)
    width = 0.35
    ind = 0
    for role, d in role_ess.items():
        p1 = plt.bar(ind, d[0], width, color = 'green')
        p2 = plt.bar(ind, d[1], width, bottom=d[0],  color = 'yellow')
        p3 = plt.bar(ind, d[2], width, bottom=d[0]+d[1],  color = 'orange')
        p4 = plt.bar(ind, d[3], width, bottom=d[0]+d[1]+d[2],  color = 'red')
        ind += 1
        
    plt.ylabel('Number of Reactions')
    plt.title(f'Reactions by RolX-role and FBA-essentiality ({model})')
    plt.xticks(np.arange(r), role_ess.keys())
    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('no damage', 'mild change', 'severe change', 'lethal'))
    plt.savefig(f'../figures/{model}_rolx.png')



models = ['BT-549', 'HCT-116', 'K-562', 'MCF7', 'OVCAR-5']

dfs = {}


for model in models:
    
    dfs[model] = runRolX(model)
    

# enables running from command line    
# if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-m', '--model', required=True)
#     io_args = parser.parse_args()
#     model = io_args.model

#     runRolX(model)
    