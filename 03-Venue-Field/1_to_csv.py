import json
from collections import namedtuple
from unicodedata import name
import pandas as pd
import csv
from tqdm import tqdm
import torch
import dgl
import os
import numpy as np

NOTE_PATH = '/nfs3-p1/zty/2021-FanXing/Academic-Networks/Academic_GNN_Module/'
DBLP_PATH = NOTE_PATH + 'dblpv13/'
"""
{'_id': '53e99784b7602d9701f3e151', 
'year': 1993, 
'venue': {'_id': '53a72a4920f7420be8bfa51b', 
    'name_d': 'International Conference on Document Analysis and Recognition', 
    'type': 0, 
    'raw': 'ICDAR-1'}, 
'fos': ['Intelligent character recognition', 'Pattern recognition', 'Computer science', 
    'Feature (computer vision)', 'Document processing', 'Handwriting recognition', 
    'Optical character recognition', 'Feature extraction', 'Feature (machine learning)', 
    'Artificial intelligence', 'Intelligent word recognition'], 
'references': ['53e99cf5b7602d97025ace63', '557e8a7a6fee0fe990caa63d', '53e9a96cb7602d97032c459a', 
    '53e9b929b7602d9704515791', '557e59ebf6678c77ea222447']}
"""
def to_csv():
    Publish = namedtuple('Publish', ['vid', 'pid', 'year', 'name_d', 'type', 'raw'])
    Field = namedtuple('Field', ['fid', 'vid', 'year'])
    Cites = namedtuple('Cites', ['pid1', 'pid2', 'year'])
    publishs, fields, cites = [], [], []

    fr = open('../save3/dblpv13.filter2.json', 'r')
    bar = tqdm(fr)
    for line in tqdm(fr):
        dic = json.loads(line)
        pid, year, venue, fos, references = \
            dic['_id'], dic['year'], dic['venue'], dic['fos'], dic['references']

        if '_id' in venue:
            vid = venue['_id']
            if 'name_d' not in venue:
                venue['name_d'] = ''
            if 'type' not in venue:
                venue['type'] = ''
            if 'raw' not in venue:
                venue['raw']  = ''
            publish = Publish(vid, pid, year, venue['name_d'], venue['type'], venue['raw'])
            publishs.append(publish)

            if fos != '[]':
                for fid in fos:
                    field = Field(fid, vid, year)
                    fields.append(field)

        if references != '[]':
            for pid2 in references:
                cite = Cites(pid, pid2, year)
                cites.append(cite)
        # bar.set_postfix(v=len(publishs), f=len(fields), c=len(cites))
    fr.close()
    print('done1')

    vdf = pd.DataFrame(columns=['vid', 'pid', 'year', 'name_d', 'type', 'raw'])
    vdf['vid'] = [v.vid for v in publishs]
    vdf['pid'] = [v.pid for v in publishs]
    vdf['year'] = [v.year for v in publishs]
    vdf['name_d'] = [v.name_d for v in publishs]
    vdf['type'] = [v.type for v in publishs]
    vdf['raw'] = [v.raw for v in publishs]
    print('done2')

    fdf = pd.DataFrame(columns=['fid', 'vid', 'year'])
    fdf['fid'] = [v.fid for v in fields]
    fdf['vid'] = [v.vid for v in fields]
    fdf['year'] = [v.year for v in fields]
    print('done3')

    cdf = pd.DataFrame(columns=['pid1', 'pid2', 'year'])
    cdf['pid1'] = [v.pid1 for v in cites]
    cdf['pid2'] = [v.pid2 for v in cites]
    cdf['year'] = [v.year for v in cites]
    print('done4')

    vdf.sort_values(by='year').to_csv(f'../save3/dblp.venue.csv', index=None, quoting=csv.QUOTE_NONNUMERIC)
    print('done5')
    fdf.sort_values(by='year').to_csv(f'../save3/dblp.field.csv', index=None, quoting=csv.QUOTE_NONNUMERIC)
    print('done6')
    cdf.sort_values(by='year').to_csv(f'../save3/dblp.cites.csv', index=None, quoting=csv.QUOTE_NONNUMERIC)
    print('done7')

def bulid_graph(path='../save3/graph_vfc.graph'):
    if os.path.exists(path):
        g_list, _ = dgl.load_graphs(path)
        return g_list[0]

    pid1 = np.load('../save2/pid1.npy')
    pid2 = np.load('../save2/pid2.npy')
    year_cite = np.load('../save2/year_cite.npy')
    vid_pub = np.load('../save2/vid_pub.npy')
    pid_pub = np.load('../save2/pid_pub.npy')
    year_pub = np.load('../save2/year_pub.npy')
    fid_has = np.load('../save2/fid_has.npy')
    vid_has = np.load('../save2/vid_has.npy')
    year_has = np.load('../save2/year_has.npy')

    idx = np.where(pid2>=0)[0]
    p1, p2 = pid1[idx], pid2[idx]
    year_cite_new = year_cite[idx]

    graph_list, _ = dgl.load_graphs(f'{DBLP_PATH}/dblp.graph')
    graph = graph_list[0]

    pid, aid = graph.edges(etype='written', order='eid')
    g1 = dgl.node_type_subgraph(graph, ['author', 'org'])
    oid_org, aid_org = g1.edges(etype='contain', order='eid')

    g2 = dgl.node_type_subgraph(graph, ['country', 'org'])
    cid_country, oid_country = g2.edges(etype='contain', order='eid')

    g3 = dgl.node_type_subgraph(graph, ['topic', 'paper'])
    tid_topic, pid_topic = g3.edges(etype='contain', order='eid')

    year = graph.edges['written'].data['year']
    author_org_year = g1.edges['contain'].data['year']
    paper_topic_year= g3.edges['contain'].data['year']

    graph_data = {
        ('paper', 'written', 'author'): (pid, aid),
        ('org', 'has', 'author'): (oid_org, aid_org),
        ('country', 'has', 'org'): (cid_country, oid_country),
        ('topic', 'contains', 'paper'): (tid_topic, pid_topic),
        ('paper', 'cites', 'paper'): (p1, p2),
        ('venue', 'publishes', 'paper'): (vid_pub, pid_pub),
        ('field', 'has', 'venue'): (fid_has, vid_has)
    }
    new_g = dgl.heterograph(graph_data)
    new_g.edata['year'] = {
        ('paper', 'written', 'author'): year,
        ('org', 'has', 'author'): author_org_year,
        ('topic', 'contains', 'paper'): paper_topic_year, 
        ('paper', 'cites', 'paper'): torch.from_numpy(year_cite_new), 
        ('venue', 'publishes', 'paper'): torch.from_numpy(year_pub), 
        ('field', 'has', 'venue'): torch.from_numpy(year_has) 
    }
    dgl.save_graphs(path, [new_g])
    return new_g


def show(path='../save3/graph_vfc.graph'):
    venues = pd.read_csv('../save3/dblp.venue.csv')
    fields = pd.read_csv('../save3/dblp.field.csv')
    cites = pd.read_csv('../save3/dblp.cites.csv')
    print(f'venues:{venues.describe()}')
    print(f'fields:{fields.describe()}')
    print(f'cites:{cites.describe()}')

    vids = venues['vid'].unique()
    vmap = {v:i for i,v in enumerate(vids)}
    venues['new_vid'] = venues['vid'].map(vmap)
    fields['new_vid'] = fields['vid'].map(vmap)

    pids = venues['pid'].unique()
    pmap = {p:i for i,p in enumerate(pids)}
    venues['new_pid'] = venues['pid'].map(pmap)
    cites['new_pid1'] = cites['pid1'].map(pmap)
    cites['new_pid2'] = cites['pid2'].map(pmap)

    fids = fields['fid'].unique()
    fmap = {f:i for i,f in enumerate(fids)}
    fields['new_fid'] = fields['fid'].map(fmap)

    print(f'venues:{len(vids)}, {venues.describe()}')
    print(f'fields:{len(fids)}, {fields.describe()}')
    print(f'cites:{len(pids)}, {cites.describe()}')

    values = {'new_pid1': -1, 'new_pid2':-1}
    cites = cites.fillna(value=values)

    values = {'new_vid': -1, 'new_pid':-1}
    venues = venues.fillna(value=values)

    values = {'new_fid': -1, 'new_vid':-1}
    fields = fields.fillna(value=values)

    pid1, pid2 = cites['new_pid1'].to_numpy(), cites['new_pid2'].to_numpy()
    vid_pub, pid_pub = venues['new_vid'].to_numpy(), venues['new_pid'].to_numpy()
    fid_has, vid_has = fields['new_fid'].to_numpy(), fields['new_vid'].to_numpy()

    year_cite = cites['year'].to_numpy()
    year_pub = venues['year'].to_numpy()
    year_has = fields['year'].to_numpy()

    def rm_nan(a, b, c):
        id1, id2 = np.where(a>=0)[0], np.where(b>=0)
        ind = id1 #if len(id1)<len(id2) else id2
        return a[ind].astype(np.int32), b[ind].astype(np.int32), torch.from_numpy(c[ind])

    pid1, pid2, year_cite = rm_nan(pid1, pid2, year_cite)
    vid_pub, pid_pub, year_pub = rm_nan(vid_pub, pid_pub, year_pub)
    fid_has, vid_has, year_has = rm_nan(fid_has, vid_has, year_has)
    
    print(f'pid1:{len(pid1)}, {pid1}')
    print(f'pid2:{len(pid2)}, {pid2}')
    print(f'vid_pub:{len(vid_pub)}, {vid_pub}')
    print(f'pid_pub:{len(pid_pub)}, {pid_pub}')
    print(f'fid_has:{len(fid_has)}, {fid_has}')
    print(f'vid_has:{len(vid_has)}, {vid_has}')

    print(f'year_cite:{len(year_cite)}, {year_cite}')
    print(f'year_pub:{len(year_pub)}, {year_pub}')
    print(f'year_has:{len(year_has)}, {year_has}')

    
    np.save('../save2/pid1.npy', pid1)
    np.save('../save2/pid2.npy', pid2)
    np.save('../save2/year_cite.npy', year_cite)
    np.save('../save2/vid_pub.npy', vid_pub)
    np.save('../save2/pid_pub.npy', pid_pub)
    np.save('../save2/year_pub.npy', year_pub)
    np.save('../save2/fid_has.npy', fid_has)
    np.save('../save2/vid_has.npy', vid_has)
    np.save('../save2/year_has.npy', year_has)

    graph_list, _ = dgl.load_graphs(f'{DBLP_PATH}/dblp.graph')
    graph = graph_list[0]

    pid, aid = graph.edges(etype='written', order='eid')
    g1 = dgl.node_type_subgraph(graph, ['author', 'org'])
    oid_org, aid_org = g1.edges(etype='contain', order='eid')

    g2 = dgl.node_type_subgraph(graph, ['country', 'org'])
    cid_country, oid_country = g2.edges(etype='contain', order='eid')

    g3 = dgl.node_type_subgraph(graph, ['topic', 'paper'])
    tid_topic, pid_topic = g3.edges(etype='contain', order='eid')

    year = graph.edges['written'].data['year']
    author_org_year = g1.edges['contain'].data['year']
    paper_topic_year= g3.edges['contain'].data['year']

    def rm_non(l):
        out = []
        for a in l:
            idx = torch.where(a>=0)[0]
            out.append(a[idx])
        return out

    pid, aid, aid_org, oid_org, oid_country, cid_country, pid_topic, tid_topic \
         = rm_non([pid, aid, aid_org, oid_org, oid_country, cid_country, 
                pid_topic, tid_topic])

    graph_data = {
        ('paper', 'written', 'author'): (pid, aid),
        ('org', 'has', 'author'): (oid_org, aid_org),
        ('country', 'has', 'org'): (cid_country, oid_country),
        ('topic', 'contains', 'paper'): (tid_topic, pid_topic),
        ('topic', 'cites', 'topic'): (pid1, pid2),
        ('venue', 'publishes', 'paper'): (vid_pub, pid_pub),
        ('field', 'has', 'venue'): (fid_has, vid_has)
    }
    new_g = dgl.heterograph(graph_data)
    new_g.edata['year'] = {
        ('paper', 'written', 'author'): year,
        ('org', 'has', 'author'): author_org_year,
        ('topic', 'contains', 'paper'): paper_topic_year, 
        ('topic', 'cites', 'topic'): year_cite, 
        ('venue', 'publishes', 'paper'): year_pub, 
        ('field', 'has', 'venue'): year_has, 
    }
    dgl.save_graphs(path, [new_g])
    print(new_g)


if __name__ =='__main__':
    # to_csv()
    bulid_graph()
    # show()
    graph_list, _ = dgl.load_graphs('../save3/graph_vfc.graph')
    graph = graph_list[0]
    print(graph)