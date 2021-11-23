import torch
import dgl
from Academic_GNN_Module.util import DBLP_PATH
from tqdm import trange

graph_list, _ = dgl.load_graphs(f'{DBLP_PATH}/dblp.graph')
graph = graph_list[0]
subgraph = graph.node_type_subgraph(['author', 'paper'])
num_paper = subgraph.number_of_nodes('paper')
freq = 50
author_ids = torch.where(subgraph.out_degrees(etype='write') >= freq)[0]
paper_ids = torch.arange(num_paper)
subg = subgraph.subgraph({'author': author_ids, 'paper': paper_ids})

coauthors = []
for year in trange(2000, 2022):
    g_list, _ = dgl.load_graphs(f'./save/coauthor_{year}.graph')
    g = g_list[0]
    g_new = dgl.node_subgraph(g, author_ids)
    coauthors.append(g_new)

new_coauthors = dgl.compact_graphs(coauthors)
dgl.save_graphs('./save2/coauthors_topic.graph', new_coauthors)
