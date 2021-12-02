from collections import namedtuple
import json
from tqdm import tqdm

"""
"authors" : [ { "_id" : "53f43b03dabfaedce555bf2a", "name" : "Min Pan" }, \
                { "_id" : "53f45ee9dabfaee43ecda842", "name" : "Chris C. N. Chu" }, \
                { "_id" : "53f42e8cdabfaee1c0a4274e", "name" : "Hai Zhou" } ], \
"""
aname = ['53f3221adabfae9a8445c0d4', '53f32b01dabfae9a8448d3fe', '53f32225dabfae9a8445c4d4', \
    '53f321c4dabfae9a8445a323', '53f321f6dabfae9a8445b41f', '53f32a1cdabfae9a844885d6', \
        '53f32208dabfae9a8445ba6a', '53f31cecdabfae9a8443f5df', '53f32a97dabfae9a8448b051', \
            '53f322bbdabfae9a8445f8ec', '53f32665dabfae9a84473b59', '53f32c82dabfae9a844956ef', \
        '53f32c9bdabfae9a84495f20', '53f32b8cdabfae9a84490408', '53f32590dabfae9a8446f3de', \
    '53f322bddabfae9a8445f96d', '53f32b5ddabfae9a8448f374', '53f32951dabfae9a84483a43', \
        '53f325b6dabfae9a844700e0', '53f325cfdabfae9a8447092a']


NOTE_PATH = '/nfs3-p1/zty/2021-FanXing/Academic-Networks/'
DBLP_PATH = NOTE_PATH + 'dblpv13/'

file_name = NOTE_PATH + 'dblp/dblpv13.compress.json'
fr = open(file_name, 'r')
# fw = open('../save3/dblpv13.filter2.json', 'w')
Author = namedtuple('Author', ['aid', 'name'])
cnt = 0
bar = tqdm(fr)
short_keys = ['_id', 'authors', 'year', 'title', 'keywords']
for line in tqdm(fr):
    dec = json.loads(line)
    flag = True
    if 'keywords' not in dec:
        dec['keywords'] = '[]'
    for k in short_keys:
        if k not in dec:
            flag = False
            break
    if not flag:
        continue

    if 'authors' in dec and len(dec['authors']) > 0:
        author_list = dec['author']
        for au in author_list:
            if 'name' not in au:
                au['name'] = ""
            author = Author(au['_id'], au['name'])

        # short = {k:dec[k] for k in short_keys}
        # ans.append(short)


        # fw.write(json.dumps(short))
        # fw.write('\n')
    # bar.set_postfix(cnt = cnt, v = len(dec['venue']))
fr.close()
# fw.close()
# print(f'cnt:{cnt}')

def to_csv():
    
    