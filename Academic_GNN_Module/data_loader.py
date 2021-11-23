'''This script works for transforming data into a unified graph.

Our DBLP data focuses on papers, which associate authors with each other.
Initially, only `Paper` has the attribute `year`, and we propagate the 
attribute to other edge relations, including `Author<-->Paper`, 
`Author<-->Org`, `Org<-->Country`, `Paper<-->Topic`.
Type: Author, has attributes `aid`, and its org and country are known
      using external links with `Org` and `Country`;
Type: Paper, has attributes `pid, title, year, keywords`, and we transform
      its title into dense vectors, label its year as edge attributes, and link
      `Paper` with top-1000 topics of each year;
Type: Org, is retrieved by matching organizations of https://csrankings.org/
      with Author's organizations in dblpv13.filter.json;
Type: Country, ir retrieved from https://csrankings.org/ with respect to the
      organization;
Type: Topic, is retrieved from `Paper`'s keywords, and we only reserve
      top-1000 topics of each year for simplicity;
Relation: Author<--write/written(Year)-->Paper;
Relation: Author<--belong/contain(Year)-->Org;
Relation: Org<--belong/contain-->Country;
Relation: Paper<--has/contain(Year)-->Topic;


```
    this.aiAreas = ["ai", "vision", "mlmining", "nlp", "ir"];
    this.systemsAreas = ["arch", "comm", "sec", "mod", "da", "bed", "hpc", "mobile", "metrics", "ops", "plan", "soft"];
    this.theoryAreas = ["act", "crypt", "log"];
    this.interdisciplinaryAreas = ["bio", "graph", "ecom", "chi", "robotics", "visualization"];
```
We have three tasks in total, 1) classifying authors into different areas,
according to classifications of https://csrankings.org/, namely `AI`, `Theory`,
`InterDisciplinary`; 2) predicting whether two authors will collaborate in the
next few years; 3) predicting the collaboration topics given two authors in
the next few years; 4) predicting the next organization an author will move to.
'''

from collections import defaultdict, namedtuple
import csv
from datetime import datetime
import math
import os
import logging

import dgl
from gensim.parsing.preprocessing import preprocess_string
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from util import set_logger, DBLP_PATH, CSRA_PATH, NOTE_PATH


def load_csv():
    '''Load data as bi-partite networks with authors and papers.

    Return
    ------
    authors : pandas.DataFrame, columns(aid, name, year, org);
    papers : pandas.DataFrame, columns(pid, year, title, keywords);
    edges : pandas.DataFrame, columns(pid, aid, year, title).
    '''
    authors = pd.read_csv(f'{DBLP_PATH}/dblp.authors.csv')
    papers = pd.read_csv(f'{DBLP_PATH}/dblp.papers.csv')
    edges = pd.read_csv(f'{DBLP_PATH}/dblp.edges.csv')
    return authors, papers, edges


def process_author_org(authors, cores=24):
    '''Standardize the unformatted author organizations.

    We have two kinds sources of academic organizations. One is from 
    CSRankings, 581 organizations with `../../CSRankings/country-info.csv` and 
    `../../CSRankings/csrankings.csv`, where the former doesn't record
    organizations of USA and the latter doesn't record the country of each
    organization. The other is from https://github.com/Hipo/university-domains-list,
    which contains 9,766 universities and their country codes. However,
    duplications of names make wrong matches between organization sources and
    author information.
    '''
    def part_process(author_part, part):
        save_path = f'{DBLP_PATH}/dblp_author_post_org/{part:02d}.csv'
        if os.path.exists(save_path):
            return pd.read_csv(save_path)

        # university = json.load(open('world_universities_and_domains.json', 'r'))
        udf = pd.read_csv(
            f'{CSRA_PATH}/country-info.csv')  # 384 organizations
        udf.columns = ['org', 'region', 'country_code']
        udf['org'] = udf['org'].apply(lambda s: s.lower())
        udf['country'] = udf['country_code'].apply(lambda s: s.upper())
        udf['nick'] = udf['org'].copy()
        # abbrv
        abbs = udf[udf['org'].apply(
            lambda s: 'university' in s)].copy().reset_index(drop=True)
        abbs['nick'] = abbs['org'].apply(
            lambda s: s.replace('university', 'univ'))
        udf = pd.concat([udf, abbs], sort=False).reset_index(drop=True)

        ranking = pd.read_csv(f'{CSRA_PATH}/csrankings.csv')
        ranking.columns = ['name', 'org', 'homepage', 'scholarid']
        us_orgs = list(set(ranking['org'].unique()) - set(udf['org']))
        us_udf = pd.DataFrame(
            columns=['nick', 'org', 'region', 'country_code'])
        us_udf['nick'] = us_orgs
        us_udf['org'] = us_orgs
        us_udf['region'] = 'North America'
        us_udf['country_code'] = 'us'
        # abbrv
        abbs = us_udf[us_udf['org'].apply(
            lambda s: 'university' in s)].copy().reset_index(drop=True)
        abbs['nick'] = abbs['org'].apply(
            lambda s: s.replace('university', 'univ'))
        us_udf = pd.concat([us_udf, abbs]).reset_index(drop=True)

        # concat non-US organizations with US organizations
        udf = pd.concat([us_udf, udf], sort=False).reset_index(drop=True)
        min_name = udf['nick'].apply(len).min()

        def uni_map(org_name):
            org_name = str(org_name).lower()
            if len(org_name) < min_name:
                return 'NOT FOUND'

            for row in udf.itertuples():
                nick, name = row.nick, row.org
                if nick in org_name:
                    return name
            return 'NOT FOUND'

        country_map = {row.org: row.country for row in udf.itertuples()}

        author_part['org'] = author_part['org'].fillna('')
        author_part['post_org'] = author_part['org'].apply(uni_map)
        author_part['country'] = author_part['post_org'].map(country_map)
        author_part.to_csv(save_path, index=None)
        return author_part

    save_dir = f'{DBLP_PATH}/dblp_author_post_org'
    save_path = f'{DBLP_PATH}/dblp.authors.postorg.csv'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(save_path):
        return pd.read_csv(save_path)

    authors = pd.read_csv(f'{DBLP_PATH}/dblp.authors.csv')
    num = len(authors)
    stride = int(math.ceil(num / cores))
    author_parts = []
    for i in range(cores):
        start = stride * i
        end = start + stride
        author_part = authors.iloc[start:end].copy()
        author_parts.append(author_part)

    ans = Parallel(n_jobs=cores,
                   verbose=30)(delayed(part_process)(author_part, i)
                               for i, author_part in enumerate(author_parts))
    authors_orgs = pd.concat(ans)
    authors_orgs.to_csv(save_path, index=None)
    return authors_orgs


def process_author_fillorg(authors, cores=24):
    '''For authors with non-empty orgs, we fill them with latest orgs.'''
    save_path = f'{DBLP_PATH}/dblp.authors.fillorg.csv'
    if os.path.exists(save_path):
        return pd.read_csv(save_path)

    authors = authors.sort_values(by='year').reset_index(drop=True)
    # authors['org'] = authors['org'].replace('', np.nan)
    ans = []
    for name, group in tqdm(authors.groupby(by=['aid'])):
        ans.append(group.fillna(method='bfill'))

    df = pd.concat(ans)
    df.to_csv(save_path, index=None)
    return df


def filter_author_with_edge_count(edges, threshold=5):
    '''Remove authors with less than `threshold` papers.'''
    cnt = edges['aid'].value_counts()
    aids = cnt[cnt >= threshold].index
    mask = edges['aid'].isin(set(aids))
    mask_edges = edges[mask].reset_index(drop=True)
    return mask_edges.copy()


def process_paper_titles(titles, cores=24):
    '''Transform a list of titles into dense vectors.'''
    def title2vec(part, words_map, word_embeds):
        ans = [] # [t1,...,tn], {w1:id}, [n+1, 300]
        for i, title in enumerate(part): # part: [t1, t2, ...]
            title = preprocess_string(title) #[w1, w2, ...]
            if len(title) == 0:
                indices = [0]
            else:
                indices = [words_map[w] for w in title] # [id1, id2]
            mean_embed = word_embeds[indices].mean(axis=0) # [300]
            ans.append(mean_embed)
        return np.stack(ans) # [n, 300]

    # transform paper title into mean embeddings
    word_name = f'{NOTE_PATH}/glove.6B.300d.txt'
    lines = open(word_name, 'r').readlines()
    split = [l.split() for l in lines] # 2*2:[ [w1,emb_300],[w2,emb_300],... ]
    words = [s[0] for s in split] # [w1, w2, ...]
    words_map = defaultdict(lambda: 0) # {w1:id}
    for i, w in enumerate(words):
        # use 0 as padding word
        words_map[w] = i + 1

    zero_padding = np.zeros((1, 300))
    embeds = np.array([s[1:] for s in split]).astype(np.float) # [n, 300]
    word_embeds = np.vstack([zero_padding, embeds]) # [n+1, 300]

    cores = 30
    stride = int(math.ceil(len(titles) / cores))
    split_titles = []
    for i in range(cores):
        start = stride * i
        end = start + stride
        part = titles[start:end]
        split_titles.append(part)
    ans = Parallel(n_jobs=30,
                   verbose=30)(delayed(title2vec)(part, words_map, word_embeds)
                               for part in split_titles)
    title_embeds = np.vstack(ans) # [n, 300]
    return title_embeds


def mine_org_country_relations(authors):
    '''Map each org to each country, including `NOT FOUND`.'''

    # Columns(aid, name, year, org, post_org, country)
    orgs = authors[['post_org', 'country']].drop_duplicates()
    # orgs['post_org'] = orgs['post_org'].replace('NOT FOUND', '')
    orgs['country'] = orgs['country'].fillna('NOT FOUND')
    org_index = {'NOT FOUND': 0}
    index = 1
    for row in tqdm(orgs.itertuples()):
        if row.post_org not in org_index:
            org_index[row.post_org] = index
            index += 1

    country_index = {'NOT FOUND': 0}
    index = 1
    for row in tqdm(orgs.itertuples()):
        if row.country not in country_index:
            country_index[row.country] = index
            index += 1
    orgs['org_id'] = orgs['post_org'].map(org_index)
    orgs['country_id'] = orgs['country'].map(country_index)
    return org_index, country_index, orgs


def mine_paper_topic_relations(papers, top=1000, cores=24):
    '''We link each paper with their keywords, where keywords are selected
    from top-1000 keywords of papers of each year.'''
    # Columns(pid, year, title, keywords)
    def split_keywords(s):
        s = s[1:-1] # "['hello', 'world']""
        s = s.replace("'", '')
        s = s.split(',')
        s = [w.lower().strip() for w in s]
        return s

    papers['keys'] = papers['keywords'].replace('[]', np.nan)
    papers = papers.dropna().reset_index(drop=True)
    papers['keys'] = papers['keywords'].apply(split_keywords) # [w1, w2, ...]

    years = papers['year'].unique()
    keywords = []
    for year in tqdm(years):
        subpapers = papers[papers['year'] == year].reset_index(drop=True)
        tmp = []
        for k in subpapers['keys']:
            if k is not None and len(k) > 0:
                tmp.extend(k)
        tmp_df = pd.Series(tmp).value_counts().iloc[:top]
        tmp_df = tmp_df.reset_index()
        tmp_df.columns = ['keyword', 'count']
        keywords.extend(tmp_df['keyword'].tolist())

    # transform papers into paper-topic links
    keywords = list(set(keywords)) 
    keywords_index = {k:i for i, k in enumerate(keywords)}
    Link = namedtuple('Link', ['pid', 'tid', 'year'])
    links = []
    for row in tqdm(papers.itertuples()):
        for k in row.keys:
            links.append(Link(row.pid, k, row.year))
    link_keys = pd.DataFrame(columns=['pid', 'key', 'year'])
    link_keys['pid'] = [link.pid for link in links]
    link_keys['key'] = [link.tid for link in links]
    link_keys['year'] = [link.year for link in links]
    # Putting `if k in keywords_index` inside for-loop over papers.itertuples()
    # yields 479.30it/s, and using map outside for-loop costs 45s in total.
    link_keys['tid'] = link_keys['key'].map(keywords_index)

    # map(keywords_index) makes 'tid' become `float` type
    link_df = link_keys.dropna().reset_index(drop=True)
    link_df = link_df[['pid', 'tid', 'year']]
    # link_df = pd.DataFrame(columns=['pid', 'tid', 'year'])
    # link_df['pid'] = [link.pid for link in links]
    # link_df['tid'] = [link.tid for link in links]
    # link_df['year'] = [link.year for link in links]

    return keywords_index, link_df # pid transform is left for main function


def build_lightgraph(authors, papers, edges, logger=None):
    '''Here we build a heterogeneous graph containing 5 node types and 8 edge
    types.
    
    Our DBLP data focuses on papers, which associate authors with each other.
    Initially, only `Paper` has the attribute `year`, and we propagate the 
    attribute to other edge relations, including `Author<-->Paper`, 
    `Author<-->Org`, `Org<-->Country`, `Paper<-->Topic`.
    Type: Author, has attributes `aid`, and its org and country are known
        using external links with `Org` and `Country`;
    Type: Paper, has attributes `pid, title, year, keywords`, and we transform
        its title into dense vectors, label its year as edge attributes, and link
        `Paper` with top-1000 topics of each year;
    Type: Org, is retrieved by matching organizations of https://csrankings.org/
        with Author's organizations in dblpv13.filter.json;
    Type: Country, ir retrieved from https://csrankings.org/ with respect to the
        organization;
    Type: Topic, is retrieved from `Paper`'s keywords, and we only reserve
        top-1000 topics of each year for simplicity;
    Relation: Author<--write/written(Year)-->Paper;
    Relation: Author<--belong/contain(Year)-->Org;
    Relation: Org<--belong/contain-->Country;
    Relation: Paper<--has/contain(Year)-->Topic;
    '''
    if logger is None:
        logger = set_logger()
    logger.info('Reindex papers and authors.')
    pids = papers['pid'].unique()
    pmap = {p: i for i, p in enumerate(pids)}
    edges['new_pid'] = edges['pid'].map(pmap)
    papers['new_pid'] = papers['pid'].map(pmap)
    aids = authors['aid'].unique()
    amap = {a: i for i, a in enumerate(aids)}
    edges['new_aid'] = edges['aid'].map(amap)
    authors['new_aid'] = authors['aid'].map(amap)
    
    logger.info('Mine Org<--->Country relations.')
    # Columns(post_org, country)
    org_index, country_index, orgs = mine_org_country_relations(authors)
    authors['post_org'] = authors['post_org'].fillna('NOT FOUND')
    authors['country'] = authors['country'].fillna('NOT FOUND')
    authors['org_id'] = authors['post_org'].map(org_index)
    authors['country_id'] = authors['country'].map(country_index)
    logger.info('Mine Paper<--->Topic relations.')
    # Columns(pid, tid, year)
    keyword_index, keyword_links = mine_paper_topic_relations(papers)
    # [['pid', 'tid', 'year']]
    keyword_links['pid'] = keyword_links['pid'].map(pmap)
    keyword_links['pid'] = keyword_links['pid'].astype(np.int64)
    keyword_links['tid'] = keyword_links['tid'].astype(np.int64)

    # Record 5 node type to index.
    logger.info('Record 5 node types to index.')
    vals = [amap, pmap, org_index, country_index, keyword_index]
    names = ['Author', 'Paper', 'Org', 'Country', 'Topic']
    dfs = []
    for name, map_vals in zip(names, vals):
        df = pd.DataFrame(columns=['Type', '_ID', 'Index'])
        df['_ID'] = list(map_vals.keys())
        df['Index'] = list(map_vals.values())
        df['Type'] = name
        dfs.append(df) 
    ids_map = pd.concat(dfs)
    ids_map.to_csv(f'{DBLP_PATH}/ids_map.csv', index=None, quoting=csv.QUOTE_NONNUMERIC)

    logger.info('Build heterogeneous graph.')
    pid = edges['new_pid'].to_numpy()
    aid = edges['new_aid'].to_numpy()
    aid_org = authors['new_aid'].to_numpy()
    oid_org = authors['org_id'].to_numpy()
    oid_country = orgs['org_id'].to_numpy()
    cid_country = orgs['country_id'].to_numpy()
    pid_topic = keyword_links['pid'].to_numpy()
    tid_topic = keyword_links['tid'].to_numpy()
    # If one columns has np.nan, dropna() makes it become np.float,
    # check each id carefully.
    for _id in [pid, aid, aid_org, oid_org, oid_country, cid_country, 
                pid_topic, tid_topic]:
        assert _id.dtype == np.int64
    
    # Relation: Author<--write/written(Year)-->Paper;
    # Relation: Author<--belong/contain(Year)-->Org;
    # Relation: Org<--belong/contain-->Country;
    # Relation: Paper<--belong/contain(Year)-->Topic;
    graph_data = {
        ('author', 'write', 'paper'): (aid, pid),
        ('paper', 'written', 'author'): (pid, aid),
        ('author', 'belong', 'org'): (aid_org, oid_org),
        ('org', 'contain', 'author'): (oid_org, aid_org),
        ('org', 'belong', 'country'): (oid_country, cid_country),
        ('country', 'contain', 'org'): (cid_country, oid_country),
        ('paper', 'belong', 'topic'): (pid_topic, tid_topic),
        ('topic', 'contain', 'paper'): (tid_topic, pid_topic)
    }
    g = dgl.heterograph(graph_data)
    year = torch.from_numpy(edges['year'].to_numpy())
    author_org_year = torch.from_numpy(authors['year'].to_numpy())
    paper_topic_year = torch.from_numpy(keyword_links['year'].to_numpy())
    g.edata['year'] = {
        ('paper', 'written', 'author'): year,
        ('author', 'write', 'paper'): year,
        ('author', 'belong', 'org'): author_org_year,
        ('org', 'contain', 'author'): author_org_year,
        ('paper', 'belong', 'topic'): paper_topic_year,
        ('topic', 'contain', 'paper'): paper_topic_year
    }
    logger.info('Done.')
    return g


if __name__ == '__main__':
    cores = 24
    logger = set_logger()

    logger.info('Load authors, papers, edges.')
    authors, papers, edges = load_csv()
    logger.info('Author with year %d, Paper %d, Edge %d.', len(authors), len(papers), len(edges))

    logger.info('Standardize author organizations.')
    authors_postorg = process_author_org(authors, cores=cores)
    logger.info('Backfill authors\' orgs one by one.')
    authors_postorg = process_author_fillorg(authors_postorg, cores=cores)
    logger.info('Author processing done.')

    logger.info('Transform paper titles into mean embeddings.')
    title_path = 'cache/dblp.papers.title.npy'
    if os.path.exists(title_path):
        title_embeds = np.load(title_path)
    else:
        title_embeds = process_paper_titles(papers['title'].tolist(), cores=cores)
        np.save(title_path, title_embeds)
    logger.info('Title embeds: {}'.format(title_embeds.shape)) # [n, 300]
    
    logger.info('Build a heterogeneous graph with (Author, Paper, Org, Country, Topic).')
    graph = build_lightgraph(authors_postorg, papers, edges)
    logger.info('Graph %s.', str(graph))
    graph_path = f'{DBLP_PATH}/dblp.graph'
    dgl.save_graphs(graph_path, [graph])