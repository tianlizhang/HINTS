import ujson as json
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from collections import namedtuple
import csv
from datetime import datetime
import logging

def set_logger():
    # set up logger
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    return logger

def to_csv(path='dblpv12/', name='dblpv12.filter.json'):
    logger = set_logger()

    fr = open(path + name, 'r')
    # logger.info('Loading json file.')
    # # records = [json.loads(line) for line in fr]
    # logger.info('Loading done.')

    Paper = namedtuple('Paper', ['pid', 'year', 'title', 'keywords'])
    Author = namedtuple('Author', ['aid', 'name', 'year', 'org'])
    Edge = namedtuple('Edge', ['pid', 'aid', 'year', 'title'])

    papers = []
    authors = []
    edges = []

    for line in tqdm(fr):
        record = json.loads(line)
        # 标题中可能存在特殊字符使得pandas解析有问题
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
    
    print('papers: {}, authors: {}, edges: {}'.format(len(papers), len(authors), len(edges)))

    edf = pd.DataFrame(columns=['pid', 'aid', 'year'])
    edf['pid'] = [e.pid for e in edges]
    edf['aid'] = [e.aid for e in edges]
    edf['year'] = [e.year for e in edges]
    edf['title'] = [e.title for e in edges]


    orig_num = len(authors)
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

    edf.sort_values(by='year').to_csv(f'{path}/dblp.edges.csv', index=None, quoting=csv.QUOTE_NONNUMERIC)
    adf.sort_values(by='year').to_csv(f'{path}/dblp.authors.csv', index=None, quoting=csv.QUOTE_NONNUMERIC)
    pdf.sort_values(by='year').to_csv(f'{path}/dblp.papers.csv', index=None, quoting=csv.QUOTE_NONNUMERIC)

    print('Done!')

if __name__ == '__main__':
    # to_csv('dblpv12/', 'dblpv12.filter.json')
    to_csv('dblpv13/', 'dblpv13.filter.json')