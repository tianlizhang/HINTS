
import dgl
import torch
from tqdm import trange, tqdm
    
import pandas as pd
import numpy as np

graph_list, _ = dgl.load_graphs('../save3/graph_vfc.graph')
graph = graph_list[0]

author_ids = torch.where(graph.in_degrees(etype='written') >= 50)[0]
paper_ids = torch.arange(graph.number_of_nodes('paper'))
topic_ids = torch.arange(graph.number_of_nodes('topic'))
venue_ids = torch.arange(graph.number_of_nodes('venue'))
field_ids = torch.arange(graph.number_of_nodes('field'))
org_ids = torch.arange(graph.number_of_nodes('org'))
country_ids = torch.arange(graph.number_of_nodes('country'))

g = graph.subgraph({'author': author_ids, 'paper': paper_ids, 'topic':topic_ids, \
    'venue': venue_ids, 'field': field_ids, 'org':org_ids, 'country': country_ids})

g = dgl.node_type_subgraph(g, ['paper', 'venue', 'author', 'topic', 'field'])


cite_year = g.edges['cites'].data['year']
df_cite = pd.DataFrame({'year': np.array(cite_year)})

id1 = torch.where(cite_year>2021)[0]
id2 = torch.where(cite_year<100)[0]
print(len(id1), len(id2))

ax = df_cite.plot(kind='kde')
# plt.xlabel('sda')
fig = ax.get_figure()
fig.savefig('fig.jpg')