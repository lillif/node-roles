import pandas as pd


e = pd.read_csv('BT-549edges.csv')

nodes = list(e['Source']) + list(e['Target'])
nodes = list(set(nodes))
nodes.sort()

idx_dict = {}

for i in range(len(nodes)):
    idx_dict[i] = nodes[i]
    
for new, old in idx_dict.items():
    e = e.replace(old, new)
    
e = e[['Source', 'Target']]
e.to_csv('rolx_edges.csv', index=False)