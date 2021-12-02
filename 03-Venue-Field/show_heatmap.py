import dgl
import torch
import numpy as np
import seaborn as sns
import pandas as pd
sns.set_theme()

NOTE_PATH = '/nfs3-p1/zty/2021-FanXing/Academic-Networks/Academic_GNN_Module/'
DBLP_PATH = NOTE_PATH + 'dblpv13/'



graph_list, _ = dgl.load_graphs(f'../save4/g_topic.graph')

year = 2020
g1 = graph_list[year-2000]

eids = g1.filter_edges(lambda x: x.data['h_index'].squeeze(1)>=70)
g2 = g1.edge_subgraph(eids)

h_ind = g2.edata['h_index'].squeeze(1)
src, dst = g2.edges(order='eid')

num=g2.num_nodes()
adj = np.zeros((num, num))
for i in range(num):
    idx = torch.where(src==i)[0]
    dst_ids = dst[idx]
    adj[i, dst_ids] = h_ind[idx]

nadj = np.array(adj)

df = pd.read_csv(f'{DBLP_PATH}/ids_map.csv')
top_ids = g2.ndata['_ID']
df2 = df[df['Type']=='Topic']
top_names = []
id2name = {}
for i in range(len(top_ids)):
    a = int(top_ids[i])
    df3 = df2[df2['Index']==a]
    ss = df3['_ID'].iloc[0]
    top_names.append((a, ss))
    id2name[i] = ss

out1, out2 = [], []
for s in top_names:
    out1.extend([s[1]]*num)
    out2.extend([s[1] for s in top_names])

df_temp = pd.DataFrame(columns=['topic1', 'topic2', 'h_index'])
df_temp['topic1'] = out1
df_temp['topic2'] = out2
df_temp['h_index'] = nadj.flatten()

target = []
tar_range = range(45, 65)

for i in tar_range:
    temp = [i*num+j for j in tar_range]
    target.extend(temp)
df_temp2 = df_temp.iloc[target]

df_pivot = df_temp2.pivot('topic1', 'topic2', 'h_index')
ax = sns.heatmap(df_pivot, cmap="YlGnBu")

# ax = sns.heatmap(nadj)
# ax = sns.heatmap(nadj[25:85, 25:85])
fig = ax.get_figure()
fig.savefig('../jpg3/heatmap2.jpg')