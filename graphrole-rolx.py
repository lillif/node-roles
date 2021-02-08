import pandas as pd
import networkx as nx

import numpy as np


from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sns


from graphrole import RecursiveFeatureExtractor, RoleExtractor

### following dkaslovsky's example from
### https://github.com/dkaslovsky/GraphRole/blob/master/examples/example.ipynb

e = pd.read_csv('BT-549edges.csv')
G = nx.DiGraph()

    
for row in e.iterrows():
    s, t, w = row[1]
    G.add_edge(s,t, weight=w)

# extract features
feature_extractor = RecursiveFeatureExtractor(G)
features = feature_extractor.extract_features()


print(f'\nFeatures extracted from {feature_extractor.generation_count} recursive generations:')
print(features)

# assign node roles
role_extractor = RoleExtractor(n_roles=None)
role_extractor.extract_role_factors(features)
node_roles = role_extractor.roles

print('\nNode role assignments:')
pprint(node_roles)

print('\nNode role membership by percentage:')
print(role_extractor.role_percentage.round(2))


# build color palette for plotting
unique_roles = sorted(set(node_roles.values()))
color_map = sns.color_palette('Paired', n_colors=len(unique_roles))
# map roles to colors
role_colors = {role: color_map[i] for i, role in enumerate(unique_roles)}
# build list of colors for all nodes in G
node_colors = [role_colors[node_roles[node]] for node in G.nodes]


pos=nx.kamada_kawai_layout(G)
# plot graph
plt.figure(figsize=[13, 10])
nx.draw(G, pos=pos, with_labels=False, 
        node_color=node_colors, node_size=80, edge_color="grey", width = 0.5)
plt.show()

# export as csv for gephi

# n = pd.read_csv('BT-549nodes.csv')
# nr_df = list(node_roles.values())
# n['Role'] = nr_df
# n.to_csv('rolx_nodes.csv', index=False)


# plotting roles & essentiality
n = pd.read_csv('BT-549nodes.csv')
essentialities = {}
for i, e in zip(n['id'], n['essentiality']):
    essentialities[i] = e
    if e < 1e-14:
        essentialities[i] = 0.0
    

role_ess = {}
for role in unique_roles:
    role_ess[role] = [0,0,0,0]

for r, e in essentialities.items():
    if e == 1.0:
        role_ess[node_roles[r]][3] += 1
    elif e > 0.5:
        role_ess[node_roles[r]][2] += 1
    elif e > 0.1:
        role_ess[node_roles[r]][1] += 1
    else:
        role_ess[node_roles[r]][0] += 1
    
role0 = role_ess['role_0']
role1 = role_ess['role_1']
role2 = role_ess['role_2']
role3 = role_ess['role_3']

ind = np.arange(4)
    
bottom = [r0 + r1 for r0, r1 in zip(role0, role1)]
bottom1 = [b + r2 for b, r2 in zip(bottom, role2)]

width = 0.35

p1 = plt.bar(ind, role0, width, color = 'green')
p2 = plt.bar(ind, role1, width, bottom=role0,  color = 'yellow')
p3 = plt.bar(ind, role2, width, bottom=bottom,  color = 'orange')
p4 = plt.bar(ind, role3, width, bottom=bottom1,  color = 'red')
    
plt.ylabel('Number of Reactions')
plt.title('Reactions by node role and essentiality (weighted graph)')
plt.xticks(ind, ('role 0', 'role 1', 'role 2', 'role 3'))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('no damage', 'mild change', 'severe change', 'lethal'))

plt.show()


    