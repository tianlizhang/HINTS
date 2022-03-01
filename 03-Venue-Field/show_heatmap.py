import dgl
import dgl.function as fn
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

import sys
sys.path.append('../')
from util import myout
sns.set_theme()

NOTE_PATH = '/nfs3-p1/zty/2021-FanXing/Academic-Networks/Academic_GNN_Module/'
DBLP_PATH = NOTE_PATH + 'dblpv13/'

def get_h(g1, df2, topk=10):
    h_all = g1.ndata['max_h'].squeeze(1)
    # cut_idx = torch.argsort(h_all)[-topk]
    # cut_h = h_all[cut_idx]

    top_idx = torch.argsort(h_all)[-topk:]
    nids = g1.nodes()[top_idx]

    # nids = g1.filter_nodes(lambda x: x.data['max_h'].squeeze(1)>=cut_h)
    g2 = dgl.node_subgraph(g1, nids)
    h_ind = g2.edata['h_index'].squeeze(1)
    src, dst = g2.edges(order='eid')

    num = g2.num_nodes()
    adj = np.zeros((num, num))
    for i in range(num):
        idx = torch.where(src==i)[0]
        dst_ids = dst[idx]
        adj[i, dst_ids] = h_ind[idx]
    
    top_ids = g2.ndata['_ID']
    top_names = []
    for i in range(len(top_ids)):
        a = int(top_ids[i])
        df3 = df2[df2['Index']==a]
        ss = df3['_ID'].iloc[0]
        top_names.append((a, ss))
    # myout(cut_h, num, adj, top_names)
    return np.array(adj), top_names


def draw_heat(nadj, top_names, path='../jpg3/heatmap2.jpg'):
    num = len(nadj)
    out1, out2 = [], []
    for s in top_names:
        out1.extend([s[1]]*num)
        out2.extend([s[1] for s in top_names])

    df_temp = pd.DataFrame(columns=['topic1', 'topic2', 'h_index'])
    df_temp['topic1'] = out1
    df_temp['topic2'] = out2
    df_temp['h_index'] = nadj.flatten()
    df_pivot = df_temp.pivot('topic1', 'topic2', 'h_index')

    fig, ax = plt.subplots(figsize=(9,9))
    ax = sns.heatmap(df_pivot, cmap="YlGnBu", square=True, annot=True, fmt='.0f')
    fig = ax.get_figure()
    fig.savefig(path, bbox_inches='tight')


if __name__ == '__main__':
    graph_list, _ = dgl.load_graphs(f'../save4/g_topic_filter.graph')
    df = pd.read_csv(f'{DBLP_PATH}/ids_map.csv')
    df2 = df[df['Type']=='Topic']

    topk_choices = [5, 7, 10]
    for year in trange(2000, 2022):
        g1 = graph_list[year-2000]
        g1.update_all(fn.copy_e('h_index', 'h'), fn.max('h', 'max_h'))
        for topk in topk_choices:
            nadj, top_names = get_h(g1, df2, topk)
            draw_heat(nadj, top_names, path=f'../jpg-5710-filter/heatmap_{topk}_{year}.jpg')



    # topk_choices = [5, 7, 10]
    # year = 2000
    # g1 = graph_list[year-2000]
    # g1.update_all(fn.copy_e('h_index', 'h'), fn.max('h', 'max_h'))
    # for topk in topk_choices:
    #     nadj, top_names = get_h(g1, df2, topk)
    #     draw_heat(nadj, top_names, path=f'../save-map/_topic_filter_2020_{topk}.jpg')
