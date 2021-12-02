import dgl
import torch
import pandas as pd
from tqdm import tqdm
import json

NOTE_PATH = '/nfs3-p1/zty/2021-FanXing/Academic-Networks/Academic_GNN_Module/'
DBLP_PATH = NOTE_PATH + 'dblpv13/'


def show_author(graph, df, df_authors, num=20):
    h_author = graph.nodes['author'].data['h_index']
    sort_idx = torch.argsort(h_author, descending=True)
    top_author = sort_idx[:num]

    df2 = df[df['Type']=='Author']
    top_author_id = []
    for i in range(len(top_author)):
        a = int(top_author[i])
        df3 = df2[df2['Index']==a]
        ss = df3['_ID'].iloc[0]
        top_author_id.append(ss)
    
    out = []
    for i, aid in enumerate(top_author_id):
        df2 = df_authors[df_authors['aid']==aid]
        ss = df2['name'].iloc[0]
        # print(ss, int(h_author[top20_author[i]]))
        out.append(  (ss, int(h_author[top_author[i]]))  )
    return out

def show_topic(graph, df, num=20):
    h_topic = graph.nodes['topic'].data['h_index']
    sort_idx = torch.argsort(h_topic, descending=True)
    top_topic = sort_idx[:num]

    df2 = df[df['Type']=='Topic']
    out = []
    for i in range(len(top_topic)):
        a = int(top_topic[i])
        df3 = df2[df2['Index']==a]
        ss = df3['_ID'].iloc[0]
        # print(f'{ss}: {h_topic[a]}')
        out.append( (ss, int(h_topic[a]))   )
    return out

def show_venue(graph, df_venue, num=20):
    h_venue = graph.nodes['venue'].data['h_index']
    sort_idx = torch.argsort(h_venue, descending=True)
    top_venue = sort_idx[:num]

    vids = df_venue['vid'].unique()
    top_vid = vids[top_venue]
    out = []
    for i in range(len(top_vid)):
        a = top_vid[i]
        df2 = df_venue[df_venue['vid']==a]
        ss = df2['name_d'].iloc[0]
        # print(f'{ss}: {h_venue[top_venue[i]]}')
        out.append( (ss, int(h_venue[top_venue[i]]) ) )
    return out

if __name__ == '__main__':
    df = pd.read_csv(f'{DBLP_PATH}/ids_map.csv')
    df_authors = pd.read_csv(f'{DBLP_PATH}/dblp.authors.csv')
    df_venue = pd.read_csv(f'../save3/dblp.venue.csv')

    graph_list, _ = dgl.load_graphs('../save3/year_graphs_h_index.graph')
    num = 20

    out = {}
    for i, graph in enumerate(tqdm(graph_list)):
        temp = {}
        year = i+2000
        out_author = show_author(graph, df, df_authors, num)
        out_topic = show_topic(graph, df, num)
        out_venue = show_venue(graph, df_venue, num)

        temp['author'] = out_author
        temp['topic'] = out_topic
        temp['venue'] = out_venue
        out[year] = temp
        if i>3:
            json.dump(temp, open(f'../save3/temp_{year}.json', 'w'))
    
    json.dump(out, open('../save3/top_atv.json', 'w'))