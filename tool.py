from tqdm import trange

NOTE_PATH = '/nfs3-p1/zty/2021-FanXing/Academic-Networks/Academic_GNN_Module/'
DBLP_PATH = NOTE_PATH + 'dblpv13/'
import pandas as pd


df = pd.read_csv(f'{DBLP_PATH}/ids_map.csv')
df2 = df[df['Type']=='Paper']
def nids2pids(nids):
    names = []
    for i in range(len(nids)):
        df3 = df2[df2['Index']==int(nids[i])]
        if len(df3)==1:
            names.append(df3['_ID'].iloc[0])
        elif len(df3)==0:
            names.append(-1)
        else:
            names.append(-2)

papers = pd.read_csv(f'{DBLP_PATH}/dblp.papers.csv')
def pids2pnames(pids):
    names = []
    for i in range(len(pids)):
        df3 = papers[papers['pid']==str(pids[i])]
        if len(df3)==1:
            names.append(df3['title'].iloc[0])
        elif len(df3)==0:
            names.append(-1)
        else:
            names.append(-2)
    return names


def get_pmap():
    df = pd.read_csv(f'{DBLP_PATH}/ids_map.csv')
    df2 = df[df['Type']=='Paper']
    df3 = df2[['_ID', 'Index']].set_index('_ID')
    pmap = df3.to_dict()['Index']
    return pmap

fields = pd.read_csv('./save_f2t/02_field2topic_top5000.csv')
def fids2fields(ls):
    out = []
    for i in range(len(ls)):
        df2 = fields[fields['fid']==ls[i]]
        if len(df2)>0:
            out.append(df2['field'].iloc[0])
        else:
            out.append('')
    return out


df = pd.read_csv(f'./save_f2t/02_field2topic_top5000.csv')
def tnames2tids(ls):
    out = []
    for i in range(len(ls)):
        df3 = df[df['topic']==str(ls[i])]
        ss = df3['tid'].iloc[0] if len(df3)>0 else -1
        out.append(ss)
    return out