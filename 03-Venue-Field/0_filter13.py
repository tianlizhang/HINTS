import json
from tqdm import tqdm


NOTE_PATH = '/nfs3-p1/zty/2021-FanXing/Academic-Networks/'
DBLP_PATH = NOTE_PATH + 'dblpv13/'

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