import dgl
from dgl.data.utils import load_info
from Academic_GNN_Module.util import DBLP_PATH
import torch
import dgl.function as fn
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed

def main():
    graph_list, _ = dgl.load_graphs(f'{DBLP_PATH}/dblp.graph')
    graph = graph_list[0]
    graph = dgl.node_type_subgraph(graph, ['author', 'paper', 'topic'])

    ids = torch.arange(10)
    paper_ids = torch.arange(10)
    topic_ids = torch.arange(10)
    # g = graph.subgraph({'author': author_ids, 'paper': paper_ids, 'topic':topic_ids})
    g = dgl.edge_subgraph(graph, {'write':ids, 'written':ids, 'contain':ids, 'belong':ids})
    print(g)
    dgl.save_graphs("save/subg1.graph", [g])

def load_data(top=1000):
    graph_list, _ = dgl.load_graphs(f'{DBLP_PATH}/dblp.graph')
    graph = graph_list[0]
    graph = dgl.node_type_subgraph(graph, ['author', 'paper', 'topic'])

    num_author, num_paper = graph.number_of_nodes('author'), graph.number_of_nodes('paper')
    graph.nodes['paper'].data['num'] = torch.ones((num_paper,1))
    graph.update_all(fn.copy_src('num', 'h'), fn.sum('h', 'num'), etype='belong')

    topic_num = graph.nodes['topic'].data['num'].squeeze(1)
    topic_idx = torch.sort(-topic_num)[1][0:top]
    
    g = graph.subgraph({'author':torch.arange(num_author), \
        'paper':torch.arange(num_paper), 'topic':topic_idx })

    topic_ids = g.ndata[dgl.NID]['topic']
    topic_map = {a: i for i, a in enumerate(list(np.array(topic_ids)))}
    topic_id = torch.tensor([topic_map[a] for a in list(np.array(topic_ids))])

    num_topic = len(topic_id)
    topic_id_ = topic_id.unsqueeze(1).clone()
    topic_onehot = torch.zeros((num_topic, num_topic)).scatter_(dim=1, index=topic_id_, value=1)
    g.nodes['topic'].data['onehot'] = topic_onehot

    g.update_all(fn.copy_src('onehot', 'h'), fn.sum('h', 'onehot'), etype='contain')
    g = dgl.node_type_subgraph(g, ['author', 'paper'])
    return g

def part_process(g, year):
    # g_list, _ = dgl.load_graphs(f"save/g_a_p.graph")
    # g = g_list[0]
    # for year in range(2021, 2022):
    write_eids = g.filter_edges(lambda x: x.data['year'] == year, etype='write')
    written_eids = g.filter_edges(lambda x: x.data['year'] == year, etype='written')

    year_graph = g.edge_subgraph({'write':write_eids, 'written': written_eids \
        }, preserve_nodes=True)
    coauthor = dgl.metapath_reachable_graph(year_graph, ['write', 'written'])
    coauthor = dgl.remove_self_loop(coauthor)

    author1, author2 = coauthor.edges(order='srcdst')

    author_src, paper_dst = year_graph.edges(etype='write')
    papers_onehot = year_graph.nodes['paper'].data['onehot']

    cnt = 0
    coauthor_label = []
    outer = tqdm(range(len(author1)))
    for i in outer:
        au1 = author1[i]
        id1 = torch.where(author_src==au1)[0]
        paper1 = paper_dst[id1]
        paper1_onehot = papers_onehot[paper1]

        au2 = author2[i]
        id2 = torch.where(author_src==au2)[0]
        paper2 = paper_dst[id2]
        paper2_onehot = papers_onehot[paper2]

        bi_paper = paper1.unsqueeze(1) - paper2
        ind1, ind2 = torch.where(bi_paper==0)

        pap1_onehot = paper1_onehot[ind1].sum(0)
        pap2_onehot = paper2_onehot[ind2].sum(0)
        pap_onehot = pap1_onehot + pap2_onehot
        coauthor_label.append(np.array(pap_onehot))

        cnt += 1
        outer.set_postfix(year=year, cnt = cnt)

    coauthor_label = torch.tensor(np.array(coauthor_label))
    coauthor_label[coauthor_label > 0] = 1
    coauthor.edata['topic'] = coauthor_label
    
    dgl.save_graphs(f"save2/coauthor_{year}.graph", [coauthor])
    return coauthor

if __name__ == '__main__':
    g = load_data()
    # dgl.save_graphs('./save2/g_author_paper.graph', [g])
    # print('end')
    coauthors = Parallel(n_jobs=22,
                   verbose=30)(delayed(part_process)(g, year)
                               for year in range(2000, 2022))
    # part_process(g)
    dgl.save_graphs("save2/coauthors.graph", coauthors)