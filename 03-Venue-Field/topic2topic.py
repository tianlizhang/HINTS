from pickle import load
import dgl
import torch
from tqdm import trange
import numpy as np
from joblib import Parallel, delayed
import pandas as pd

df_filtered = pd.read_csv('../save-map/synonym_topic_filtered.csv')
index1 = df_filtered['index1'].tolist()
index2 = df_filtered['index2'].tolist()

def filter_topic(ls):
    for i in trange(len(index1)-1, -1, -1):
        for j in range(len(ls)):
            src = ls[j]
            id1 = torch.where(src==index1[i])[0]
            if len(id1) > 0:
                src[id1] = index2[i]
            ls[j] = src
    return ls

def load_data():
    graph_list, _ = dgl.load_graphs(f'../save3/graph_vfc.graph')
    graph = graph_list[0]
    g = dgl.node_type_subgraph(graph, ['paper', 'topic'])
    print('g:', g)

    src_top, dst_pap = g.edges(etype='contains')
    src_pap1, dst_pap2 = g.edges(etype='cites')
    cite_year = g.edges['cites'].data['year']
    contains_year = g.edges['contains'].data['year']

    src_top = filter_topic([src_top])[0]
    g2 = dgl.heterograph({
        ('paper', 'cites', 'paper'): (src_pap1, dst_pap2), 
        ('topic', 'contains', 'paper'): (src_top, dst_pap), 
        ('paper', 'belongs', 'topic'): (dst_pap, src_top)
    })
    g2.edata['year'] = {
        ('paper', 'cites', 'paper'): cite_year, 
        ('topic', 'contains', 'paper'): contains_year, 
        ('paper', 'belongs', 'topic'): contains_year
    }
    print('g2:', g2)

    out = []
    outer = range(2000, 2022)
    for year in outer:
        contains_eids = g2.filter_edges(lambda x: x.data['year'] == year, etype='contains')
        belong_eids = g2.filter_edges(lambda x: x.data['year'] == year, etype='belongs')

        year_graph = g2.edge_subgraph({'contains': contains_eids, \
            'belongs': belong_eids}, preserve_nodes=True)

        g_topic = dgl.metapath_reachable_graph(year_graph, ['contains', 'belongs'])
        g_topic = dgl.remove_self_loop(g_topic)

        # topic_src, paper_dst = year_graph.edges(etype='contains')
        top1, top2 = g_topic.edges(order='srcdst')

        h_index = torch.zeros(len(top1), 1)
        bar = trange(len(top1))
        for i in bar:
            pap1 = year_graph.successors(top1[i], etype='contains')
            pap2 = year_graph.successors(top2[i], etype='contains')

            p1, p2 = set(np.array(pap1)), set(np.array(pap2))
            idx = p1 & p2

            # pap1 = pap1.unique()
            # pap2 = pap2.unique()
            # bi = pap1[:, None] - pap2
            # idx = torch.where(bi==0)[0]

            h_index[i, 0] = len(idx) if len(idx) > 0 else 0

            bar.set_postfix(year=year, h=len(idx))
        
        g_topic.edata['h_index'] = h_index
        out.append(g_topic)

        # bar.set_postfix(h=torch.max(h_index))
        
        # if year<2003:
        dgl.save_graphs(f'../save4/g_topic_filter_{year}.graph', [g_topic])
    
    dgl.save_graphs(f'../save4/g_topic_filter.graph', out)


def get_data():
    graph_list, _ = dgl.load_graphs(f'../save3/graph_vfc.graph')
    graph = graph_list[0]
    g = dgl.node_type_subgraph(graph, ['paper', 'topic'])
    # print('g:', g)

    src_top, dst_pap = g.edges(etype='contains')
    src_pap1, dst_pap2 = g.edges(etype='cites')
    cite_year = g.edges['cites'].data['year']
    contains_year = g.edges['contains'].data['year']

    src_top = filter_topic([src_top])[0]
    g2 = dgl.heterograph({
        ('paper', 'cites', 'paper'): (src_pap1, dst_pap2), 
        ('topic', 'contains', 'paper'): (src_top, dst_pap), 
        ('paper', 'belongs', 'topic'): (dst_pap, src_top)
    })
    g2.edata['year'] = {
        ('paper', 'cites', 'paper'): cite_year, 
        ('topic', 'contains', 'paper'): contains_year, 
        ('paper', 'belongs', 'topic'): contains_year
    }
    # print('g2:', g2)
    return g2

def part(g2, year):
    contains_eids = g2.filter_edges(lambda x: x.data['year'] == year, etype='contains')
    belong_eids = g2.filter_edges(lambda x: x.data['year'] == year, etype='belongs')

    year_graph = g2.edge_subgraph({'contains': contains_eids, \
        'belongs': belong_eids}, preserve_nodes=True)

    g_topic = dgl.metapath_reachable_graph(year_graph, ['contains', 'belongs'])
    g_topic = dgl.remove_self_loop(g_topic)

    # topic_src, paper_dst = year_graph.edges(etype='contains')
    top1, top2 = g_topic.edges(order='srcdst')

    h_index = torch.zeros(len(top1), 1)
    bar = range(len(top1))
    for i in bar:
        pap1 = year_graph.successors(top1[i], etype='contains')
        pap2 = year_graph.successors(top2[i], etype='contains')

        p1, p2 = set(np.array(pap1)), set(np.array(pap2))
        idx = p1 & p2

        h_index[i, 0] = len(idx) if len(idx) > 0 else 0
        # bar.set_postfix(year=year, h=len(idx))
    
    g_topic.edata['h_index'] = h_index
    dgl.save_graphs(f'../save4/g_topic_filter_{year}.graph', [g_topic])
    return g_topic

if __name__ =='__main__':
    # load_data()

    g2 = get_data()
    out = Parallel(n_jobs=22,
                   verbose=30)(delayed(part)(g2, year)
                               for year in range(2000, 2022))
    dgl.save_graphs(f'../save-map/g_topic_filter.graph', out)