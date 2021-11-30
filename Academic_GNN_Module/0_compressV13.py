import json
import re

import pandas as pd
import numpy as np
from tqdm import tqdm

def compress13():
    fr = open('../dblp/dblpv13.json', 'r')
    fw = open('../dblp/dblpv13.compress.json', 'w')
    tot = ''
    for i, line in enumerate(tqdm(fr)):
        # if i >= 10000:
        #     break
        if line.startswith('[') or line.endswith(']'):
            tot = ''
            continue
        if line.startswith('{'):
            tot = '{'
        elif line.startswith('}'):
            # 我们只保留每篇paper的始末'{}'，不保留多余的,
            tot += '}'
            tot = tot.replace('\n', '')
            tot = re.sub(r'\s+', ' ', tot)
            tot = re.sub(r'NumberInt\((\d+)\)', r'\1', tot)
            fw.write(tot.strip())
            fw.write('\n')
            tot = ''
        else:
            tot += line
    fr.close()
    fw.close()

def filter13(file_name):
    '''
    { "_id" : "53e99784b7602d9701f3e15d", \
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
        "Static timing analysis", "Statistics" ], \
    "n_citation" : 28, \
    "page_start" : "2461", \
    "page_end" : "2464Vol.3", \
    "lang" : "en", \
    "volume" : "", \
    "issue" : "", \
    "issn" : "", \
    "isbn" : "0-7803-8834-8", \e
    "doi" : "10.1109/ISCAS.2005.1465124", \
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
            short = {k:dec[k] for k in short_keys}
            ans.append(short)
            fw.write(json.dumps(short))
            fw.write('\n')

    fr.close()
    fw.close()

if __name__ == '__main__':
    # compress13()
    filter13('../dblp/dblpv13.compress.json')



