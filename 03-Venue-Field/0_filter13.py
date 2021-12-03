import json
from tqdm import tqdm
import pandas as pd

NOTE_PATH = '/nfs3-p1/zty/2021-FanXing/Academic-Networks/'
DBLP_PATH = NOTE_PATH + 'dblpv13/'

def fillter13():
    file_name = NOTE_PATH + 'dblp/dblpv13.compress.json'
    fr = open(file_name, 'r')
    fw = open('../save3/dblpv13.filter2.json', 'w')

    short_keys = ['_id', 'year', 'venue', 'fos', 'references']
    cnt = 0
    bar = tqdm(fr)
    for line in bar:
        dec = json.loads(line)
        if '_id' not in dec:
            continue
        if 'year' not in dec:
            continue
        if 'venue' not in dec:
            continue

        if 'fos' not in dec:
            dec['fos'] = '[]'
        if 'references' not in dec:
            dec['references'] = '[]'

        if len(dec['venue']) > 0:
            short = {k:dec[k] for k in short_keys}
            # ans.append(short)
            fw.write(json.dumps(short))
            fw.write('\n')
            cnt += 1
        bar.set_postfix(cnt = cnt, v = len(dec['venue']))
    fr.close()
    fw.close()
    print(f'cnt:{cnt}')


def check_turing():
    a = [('2020', 'Aho, Alfred V.', '56314f0045cedb3399d765d2', '3178630', -1, 67),
    ('2020', 'Jeffrey D. Ullman', '53f48c0fdabfaea7cd1cdf72', '1548191', 24, 117),
    ('2019', 'Edwin E. Catmull', '53f454f4dabfaec09f201277', '1152321', -1, 6),
    ('2019', 'Pat Hanrahan', '53f42ff1dabfaee4dc7395aa', '390906', 18, 93),
    ('2018', 'Yoshua Bengio', '53f4ba75dabfaed83977b7db', '1572517', 21, 196),
    ('2018', 'Geoffrey E. Hinton', '53f366a7dabfae4b3499c6fe', '57714', 19, 166),
    ('2018', 'Y. LeCun', '53f48919dabfaee4dc8b219c', '1546523', 13, 129)]

    a_ID = [i[2] for i in a]

    file_name = NOTE_PATH + 'dblp/dblpv13.compress.json'
    fr = open(file_name, 'r')

    out = [ [] for _ in range(len(a_ID))]
    bar = tqdm(fr)
    for line in bar:
        dec = json.loads(line)
        if ('_id' not in dec) or ('title' not in dec) or \
            ('authors' not in dec) or ('year' not in dec):
            continue

        pid, title, authors, year = \
            dec['_id'], dec['title'], dec['authors'], dec['year']

        for author in authors:
            if '_id' in author:
                author_id = author['_id']
                for i, idx in enumerate(a_ID):
                    if idx == author_id:
                        out[i].append((idx, pid, title, authors, year))
    
    print(out)
    json.dump(out, open('out.json', 'w')) 
    fr.close()


def check_turing2():
    a = [('2020', 'Aho, Alfred V.', '56314f0045cedb3399d765d2', '3178630', -1, 67),
    ('2020', 'Jeffrey D. Ullman', '53f48c0fdabfaea7cd1cdf72', '1548191', 24, 117),
    ('2019', 'Edwin E. Catmull', '53f454f4dabfaec09f201277', '1152321', -1, 6),
    ('2019', 'Pat Hanrahan', '53f42ff1dabfaee4dc7395aa', '390906', 18, 93),
    ('2018', 'Yoshua Bengio', '53f4ba75dabfaed83977b7db', '1572517', 21, 196),
    ('2018', 'Geoffrey E. Hinton', '53f366a7dabfae4b3499c6fe', '57714', 19, 166),
    ('2018', 'Y. LeCun', '53f48919dabfaee4dc8b219c', '1546523', 13, 129)]

    a_ID = [i[2] for i in a]

    file_name = NOTE_PATH + 'dblp/dblpv13.compress.json'
    fr = open(file_name, 'r')

    out = [ [] for _ in range(len(a_ID))]
    bar = tqdm(fr)
    for line in bar:
        dec = json.loads(line)
        if ('title' not in dec) or ('authors' not in dec) :
            continue

        title, authors = dec['title'], dec['authors']

        for author in authors:
            if '_id' in author:
                author_id = author['_id']
                for i, idx in enumerate(a_ID):
                    if idx == author_id:
                        out[i].append((idx, title, authors))
    
    # print(out)
    json.dump(out, open('../save3/out2.json', 'w'))
    fr.close()

    turing_name = [i[1] for i in a]
    for i,item in enumerate(out):
        print(turing_name[i], len(item))

    df_out = pd.DataFrame(columns=['name', 'title'])
    ll = []
    for idx in range(len(turing_name)):
        # print(turing_name[idx])
        for item in out[idx]:
            ll.append((turing_name[idx], item[1]))
    df_out['name'] = [item[0] for item in ll]
    df_out['title'] = [item[1] for item in ll]
    df_out.to_csv('../save3/turing2.csv')


if __name__ == '__main__':
    # fillter13()

    check_turing2()
