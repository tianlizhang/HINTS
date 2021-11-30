import json
import re

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import namedtuple
import csv
import dgl
import os

NOTE_PATH = '/nfs3-p1/zty/2021-FanXing/Academic-Networks/Academic_GNN_Module/'
DBLP_PATH = NOTE_PATH + 'dblpv13/'

def filter13(file_name):
    '''
    { "_id" : "53e99784b7602d9701f3e15d", \ # paper ID
    "title" : "Timing yield estimation using statistical static timing analysis", \
    "authors" : [ { "_id" : "53f43b03dabfaedce555bf2a", "name" : "Min Pan" }, \
                { "_id" : "53f45ee9dabfaee43ecda842", "name" : "Chris C. N. Chu" }, \
                { "_id" : "53f42e8cdabfaee1c0a4274e", "name" : "Hai Zhou" } ], \
    "venue" : { "_id" : "53a72e2020f7420be8c80142", \
            "name_d" : "International Symposium on Circuits and Systems", \
            "type" : 0, \
            "raw" : "ISCAS (3)" }, \
    "year" : 2005, \
    "keywords" : [ "sequential circuits", "statistical distributions", "set-up time constraints", \
        "register-to-register paths", "statistical static timing analysis", "integrated circuit modelling",\
        "parameter estimation", "statistical analysis", "circuit model", "path delays", "deep sub-micron technology",\
        "timing", "delay distributions", "delays", "circuit timing", "shortest path variations", \
        "hold time constraints", "integrated circuit yield","process variations", "integrated circuit layout",\
        "high-performance circuit designs", "clock skew", \
        "timing yield estimation", "deterministic static timing analysis", "monte carlo simulation", "design method", \
        "static timing analysis", "design methodology", "process variation", "shortest path", "registers", \
        "circuit design", "circuit analysis" ], \
    "fos" : [ "Delay calculation", "Timing failure", "Monte Carlo method", "Sequential logic", \
        "Statistical static timing analysis", "Shortest path problem", "Computer science", "Algorithm", "Clock skew", \
        "Static timing analysis", "Statistics" ], \ # paper fields of study
    "n_citation" : 28, \ #  citation number.引文编号
    "page_start" : "2461", \
    "page_end" : "2464Vol.3", \
    "lang" : "en", \ # 语言
    "volume" : "", \ # 量
    "issue" : "", \ # 期号
    "issn" : "", \ # 标准国际连续出版物号(International Standard Serial Number)的简称
    "isbn" : "0-7803-8834-8", \ # 国际标准书号（International Standard Book Number）
    "doi" : "10.1109/ISCAS.2005.1465124", \ # 数字对象唯一标识符 digital object unique identifier
    "pdf" : "//static.aminer.org/pdf/PDF/000/423/329/timing_yield_estimation_using_statistical_static_timing_analysis.pdf", \
    "url" : [ "http://dx.doi.org/10.1109/ISCAS.2005.1465124", \
        "http://ieeexplore.ieee.org/xpl/abstractAuthors.jsp?tp=&arnumber=1465124" ], \
    "abstract" : "As process variations become a significant problem in deep sub-micron technology, a shift from \
        deterministic static timing analysis to statistical static timing analysis for high-performance circuit designs \
        could reduce the excessive conservatism that is built into current timing design methods. We address the timing \
        yield problem for sequential circuits and propose a statistical approach to handle it. We consider the spatial and \
        path reconvergence correlations between path delays, set-up time and hold time constraints, and clock skew due to \
        process variations. We propose a method to get the timing yield based on the delay distributions of register-to-register\
        paths in the circuit On average, the timing yield results obtained by our approach have average errors of less than 1.0% \
        in comparison with Monte Carlo simulation. Experimental results show that shortest path variations and clock skew due \
        to process variations have considerable impact on circuit timing, which could bias the timing yield results. \
        In addition, the correlation between longest and shortest path delays is not significant.", \
    "references" : [ "53e9a8a9b7602d97031f6bb9", "599c7b6b601a182cd27360da", "53e9b443b7602d9703f3e52b", \
            "53e9a6a6b7602d9702fdc57e", "599c7b6a601a182cd2735703", "53e9aad9b7602d970345afea", \
                "5582821f0cf2bf7bae57ac18", "5e8911859fced0a24bb9a2ba", "53e9b002b7602d9703a5c932" ]}
    '''
    print(file_name)
    fr = open(file_name, 'r')
    output_name = 'dblpv13/dblpv13.filter.json'
    fw = open(output_name, 'w')
    ans = []
    short_keys = ['_id', 'authors', 'year', 'title', 'keywords', 'venue', 'references']
    for line in tqdm(fr):
        dec = json.loads(line)
        flag = True
        if 'keywords' not in dec:
            dec['keywords'] = '[]'
        if 'references' not in dec:
            dec['references'] = '[]'
        for k in short_keys:
            if k not in dec:
                flag = False
                break
        if not flag:
            continue

        if 'authors' in dec and len(dec['authors']) > 0:
            short = {k:dec[k] for k in short_keys}
            ans.append(short)
            fw.write(json.dumps(short))
            fw.write('\n')
    fr.close()
    fw.close()


def to_csv(path='dblpv12/', name='dblpv12.filter.json'):
    """
    "venue" : { "_id" : "53a72e2020f7420be8c80142", \
            "name_d" : "International Symposium on Circuits and Systems", \
            "type" : 0, \
            "raw" : "ISCAS (3)" }, 
    """
    fr = open(path + name, 'r')
    Paper = namedtuple('Paper', ['pid', 'year', 'title', 'keywords'])
    Author = namedtuple('Author', ['aid', 'name', 'year', 'org'])
    Edge = namedtuple('Edge', ['pid', 'aid', 'year', 'title'])
    Publish = namedtuple('Publish', ['vid', 'pid', 'year', 'name_d', 'type', 'raw'])
    Cites = namedtuple('Cites', ['pid1', 'pid2', 'year'])

    papers, authors, edges, publishs = [], [], [], []
    for line in tqdm(fr):
        record = json.loads(line)
        title = re.sub('[^A-Za-z0-9]+', ' ', record['title'])
        paper = Paper(record['_id'], record['year'], title, record['keywords'])
        papers.append(paper)
        for arecord in record['authors']:
            if '_id' not in arecord:
                continue
            if 'name' not in arecord:
                arecord['name'] = ''
            if 'org' not in arecord:
                arecord['org'] = ''
            author = Author(arecord['_id'], arecord['name'], paper.year, arecord['org'])
            authors.append(author)
            edge = Edge(paper.pid, author.aid, paper.year, title)
            edges.append(edge)
        
        for vrecord in record['venue']:
            if '_id' not in vrecord:
                continue
            if 'name_d' not in vrecord:
                vrecord['name_d'] = ''
            if 'type' not in vrecord:
                vrecord['type'] = ''
            if 'raw' not in vrecord:
                vrecord['raw']  = ''
            publish = Publish(vrecord['_id'], paper.pid, paper.year, vrecord['name_d'], vrecord['type'], vrecord['raw'])
            publishs.append(publish)
        
        # for crecord in record['reference']
    
    print('papers: {}, authors: {}, edges: {}, publish:{}'.format(len(papers), len(authors), len(edges), len(publishs)))

    edf = pd.DataFrame(columns=['pid', 'aid', 'year'])
    edf['pid'] = [e.pid for e in edges]
    edf['aid'] = [e.aid for e in edges]
    edf['year'] = [e.year for e in edges]
    edf['title'] = [e.title for e in edges]

    adf = pd.DataFrame(columns=['aid', 'name', 'year', 'org'])
    adf['aid'] = [a.aid for a in authors]
    adf['name'] = [a.name for a in authors]
    adf['year'] = [a.year for a in authors]
    adf['org'] = [a.org for a in authors]
    adf.drop_duplicates(subset=['aid', 'year'], inplace=True)
    print('authors are merged to {}'.format(len(adf)))

    pdf = pd.DataFrame(columns=['pid', 'year', 'title', 'keywords'])
    pdf['pid'] = [p.pid for p in papers]
    pdf['year'] = [p.year for p in papers]
    pdf['title'] = [p.title for p in papers]
    pdf['keywords'] = [p.keywords for p in papers]

    vdf = pd.DataFrame(columns=['vid', 'pid', 'year', 'name_d', 'type', 'raw'])
    vdf['vid'] = [v.vid for v in publishs]
    vdf['pid'] = [v.pid for v in publishs]
    vdf['year'] = [v.year for v in publishs]
    vdf['name_d'] = [v.name_d for v in publishs]
    vdf['type'] = [v.type for v in publishs]
    vdf['raw'] = [v.raw for v in publishs]

    edf.sort_values(by='year').to_csv(f'{path}/dblp.edges.csv', index=None, quoting=csv.QUOTE_NONNUMERIC)
    adf.sort_values(by='year').to_csv(f'{path}/dblp.authors.csv', index=None, quoting=csv.QUOTE_NONNUMERIC)
    pdf.sort_values(by='year').to_csv(f'{path}/dblp.papers.csv', index=None, quoting=csv.QUOTE_NONNUMERIC)
    vdf.sort_values(by='year').to_csv(f'{path}/dblp.venues.csv', index=None, quoting=csv.QUOTE_NONNUMERIC)
    print('Done!')


def build_graph(authors, papers, edges, venues):
    pids = papers['pid'].unique()
    pmap = {p:i for i,p in enumerate(pids)}
    papers['new_pid'] = papers['pid'].map(pmap)
    edges['new_pid'] = edges['pid'].map(pmap)
    venues['new_pid'] = venues['pid'].map(pmap)

    aids = authors['aid'].unique()
    amap = {a:i for i,a in enumerate(aids)}
    authors['new_aid'] = authors['aid'].map(amap)
    edges['new_aid'] = edges['aid'].map(amap)

    vids = venues['vid'].unique()
    vmap = {v:i for i,v in enumerate(vids)}
    venues['new_vid'] = venues['vid'].map(vmap)

    aid = edges['new_aid'].to_numpy()
    pid = edges['new_pid'].to_numpy()

    
def bulid_new_g(path='../save3/new_g.graph'):
    if os.path.exists(path):
        g_list, _ = dgl.load_graphs(path)
        return g_list[0]

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
        ('topic', 'contains', 'paper'): (tid_topic, pid_topic)
    }
    new_g = dgl.heterograph(graph_data)
    new_g.edata['year'] = {
        ('paper', 'written', 'author'): year,
        ('org', 'has', 'author'): author_org_year,
        ('topic', 'contains', 'paper'): paper_topic_year
    }
    dgl.save_graphs(path, [new_g])
    return new_g

if __name__ == '__main__':
    # compress13()
    filter13('../dblp/dblpv13.compress.json')
    to_csv('dblpv13/', 'dblpv13.filter.json')
    '''
    authors : pandas.DataFrame, columns(aid, name, year, org);
    papers : pandas.DataFrame, columns(pid, year, title, keywords);
    edges : pandas.DataFrame, columns(pid, aid, year, title).
    venues : pandas.DataFrame, columns(vid, pid, year, name_d, type, raw)
    '''
    DBLP_PATH = './'
    authors = pd.read_csv(f'{DBLP_PATH}/dblp.authors.csv')
    papers = pd.read_csv(f'{DBLP_PATH}/dblp.papers.csv')
    edges = pd.read_csv(f'{DBLP_PATH}/dblp.edges.csv')
    venues = pd.read_csv(f'{DBLP_PATH}/venues.csv')





