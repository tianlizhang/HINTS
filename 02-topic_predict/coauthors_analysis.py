import networkx as nx
from numpy.core.defchararray import title
import torch
import dgl
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np
from random import randint
import pandas as pd


def draw_spl(shortest_path_len, path=''):
    path_lengths = np.zeros(100, dtype=int)
    for pls in shortest_path_len.values():
        pl, cnts = np.unique(list(pls.values()), return_counts=True)
        path_lengths[pl] += cnts
    idx = np.argmax(path_lengths==0)
    path_lengths = path_lengths[:idx]
    
    freq_percent = 100 * path_lengths[1:] / path_lengths[1:].sum()

    plt.figure(figsize=(15,8))
    plt.bar(np.arange(1, len(path_lengths)), height=freq_percent)
    plt.title('Distribution of shortest path length in G')
    plt.xlabel('Shortest Path Length')
    plt.ylabel('Frequency (%)')
    plt.savefig(path)


def draw_hist(centrality, path='', title='', bins=100):
    plt.figure(figsize=(15,8))
    plt.hist(centrality.values(), bins=bins)
    plt.title(title + ' Histogram ', fontdict ={'size': 35}, loc='center') 
    plt.xlabel(title, fontdict ={'size': 20})
    plt.ylabel('Counts',fontdict ={'size': 20})
    if path!='':
        plt.savefig(path)


def draw_communities(G, communities, path=''):
    colors = ['' for x in range (G.number_of_nodes())]  # initialize colors list
    counter = 0
    for com in communities:
        color = '#%06X' % randint(0, 0xFFFFFF)  # creates random RGB color
        counter += 1
        for node in list(com):  # fill colors list with the particular color for the community nodes
            colors[node] = color
    counter
    plt.figure(figsize=(15,9))
    plt.axis('off') 
    nx.draw_networkx(G, pos=None, node_size=10, with_labels=False, width=0.15, node_color=colors)
    plt.savefig(path)


def analysis(G, i):
    ## 1. 基础拓扑特征
    # a. |V|, |E|, degree：平均和最大
    E = G.number_of_edges()
    V = G.number_of_nodes()
    degree = [d for _, d in G.degree()]
    d_avg = np.mean(degree)
    d_max = np.max(degree)
    # b. 最短路径：分布、平均
    shortest_path_len = dict(nx.all_pairs_shortest_path_length(G))
    l_avg = np.mean([np.mean(list(spl.values())) for spl in shortest_path_len.values()])
    draw_spl(shortest_path_len, path=f'../jpg/spl_{i}.jpg')
    # c. 密度
    ρ = nx.density(G)
    # d. 连通分量个数
    n_cc = nx.number_connected_components(G)
    # e. 社区个数：社区是一组节点，组内的节点连接的边比组之间的多得多。
    communities = nx.community.label_propagation_communities(G)
    nx.number_weakly_connected_components
    nx.number_strongly_connected_components
    n_c = len(list(communities))

    ## 2. 中心性度量
    # a. 度中心性：此节点连接到其他节点的比例
    dc = nx.centrality.degree_centrality(G)
    dc_top8 = (sorted(dc.items(), key=lambda item: item[1], reverse=True))[:8]
    draw_hist(dc, path=f'../jpg2/dc_{i}.jpg', title='Degree Centrality')
    # b. 介数中心性：一个节点位于其他节点之间最短路径上的次数百分比，这意味着它充当桥梁
    bc = nx.centrality.betweenness_centrality(G)
    bc_top8 = (sorted(bc.items(), key=lambda item: item[1], reverse=True))[:8]
    draw_hist(bc, path=f'../jpg2/bc_{i}.jpg', title='Betweenness Centrality')
    # c. 精密度中心性：与所有其他节点的平均距离。越高，离网络的中心越近。
    cc = nx.centrality.closeness_centrality(G)
    cc_top8 = (sorted(cc.items(), key=lambda item: item[1], reverse=True))[:8]
    draw_hist(cc, path=f'../jpg2/cc_{i}.jpg', title='Closeness Centrality')
    # d. 特征向量中心性: 与其他重要节点的连接程度，根据连接程度、链接数量来衡量影响力
    # if i==14 or i==18:
    #     ec=-100
    # else:
    #     ec = nx.centrality.eigenvector_centrality(G)  # save results in a variable to use again 
    #     ec_top8 = (sorted(ec.items(), key=lambda item: item[1], reverse=True))[:8]
    #     draw_hist(cc, path=f'../jpg2/ec_{i}.jpg', title='Eigenvector Centrality')

    ## 3. Triangles
    triangles = nx.triangles(G)
    T_top8 = (sorted(triangles.items(), key=lambda item: item[1], reverse=True))[:8]
    triangles_per_node = list(triangles.values())
    T = sum(triangles_per_node) / 3
    T_avg = np.mean(triangles_per_node)
    T_max = np.max(triangles_per_node)

    ## 4. Coefficient系数
    # a. Assort. Coeff. 分类系数描述了网络节点以某种方式附加到其他节点的偏好
    r = nx.degree_assortativity_coefficient(G)
    # b. 聚类系数：两个随机选择的朋友彼此成为朋友的概率，是三元闭包的标志
    K = nx.clustering(G)
    K_avg = nx.average_clustering(G)
    K_top8 = (sorted(K.items(), key=lambda item: item[1], reverse=True))[:8]
    draw_hist(K, path=f'../jpg2/K_{i}.jpg', title='Clustering Coefficient')

    ## 5. Maximum Cliques 最大团: 包含v的最大完全子图
    max_cliq = list(nx.find_cliques_recursive(G))
    w_num = len(max_cliq)
    w_lb = min([len(nc) for nc in max_cliq])
    w_hb = max([len(nc) for nc in max_cliq])

    ## 6. Bridges: 删除边会导致 A 和 B 位于不同的分量中
    bridges = list(nx.bridges(G))
    local_bridges = list(nx.local_bridges(G, with_span=False))
    n_b, n_lb = len(bridges), len(local_bridges)

    kcore = list(nx.k_core(G))
    K_max = max(kcore)

    return V, E, d_max, d_avg, l_avg, ρ, n_cc, n_c, \
        T, T_avg, T_max,\
        r, K_avg, \
        w_num, w_lb, w_hb, \
        n_b, n_lb, K_max

def main():
    coauthors, _ = dgl.load_graphs(f'../save2/coauthors_topic.graph')
    out = []
    for i in trange(19, len(coauthors)):
    # for i in trange(0, 6):
        g = coauthors[i]
        edge_topic = g.edata['topic']
        idx = torch.where(edge_topic.sum(1)!=0)[0]
        new_g = dgl.edge_subgraph(g, idx, preserve_nodes=False)
        G = dgl.to_networkx(new_g, edge_attrs=['topic'])
        G = nx.Graph(G)

        paras = analysis(G, i)
        paras = np.array(list(paras))
        print(i, paras)
        out.append(paras)

        np.save(f'../save2/paras_{i}.npy', paras)
    out = np.array(out)

    df = pd.DataFrame(out)
    df.columns = ['V', 'E', 'd_max', 'd_avg', 'l_avg', 'ρ', 'n_cc', 'n_c', \
            'T', 'T_avg', 'T_max', \
                'r', 'K_avg', \
                    'w_num', 'w_lb', 'w_hb',\
                        'n_b', 'n_lb', 'K_max']
    df.index = [str(year) for year in range(2000, 2000+len(coauthors))]
    df.to_csv('../save2/analysis.csv', index=False)


def record():
    a = [0, 1, 2,3,4,5,6,7]
    a[0] = '1.92660000e+04 2.71000000e+04 4.6000e+01 2.81324613e+00 \
        7.17188658e+08 1.46028868e-04 2.43300000e+03 4.90100000e+03 \
            1.45870000e+04 2.27141078e+00 3.810000e+02 3.89537440e-01 \
                3.45973752e-01 1.36090000e+04 2.000000e+00 2.100000e+01 \
                    6.92908080e+03 8.67500000e+03 1.85960000e+04'

    a[1] = '2.05260000e+04 3.07290000e+04 4.80000000e+01 2.99415376e+00 \
        7.24742654e+00 1.45878380e-04 2.29300000e+03 4.99300000e+03 \
            1.81340000e+04 2.65039462e+00 2.98000000e+02 3.54850864e-01 \
                3.62341985e-01 1.46390000e+04 2.00000000e+00 2.30000000e+01 \
                    6.97200000e+03 8.94400000e+03 1.87210000e+04'

    a[2] = '2.19200000e+04 3.42870000e+04 4.70000000e+01 3.12723459e+00 \
        7.38404299e+00 1.42620267e-04 2.21300000e+03 5.20800000e+03 \
            2.01810000e+04 2.76099051e+00 3.21000000e+02 3.14377346e-01 \
                3.57147666e-01 1.61880000e+04 2.0000000e+00 2.200000e+01 \
                    7.23200000e+03 9.80600000e+03 2.07670000e+04'

    a[3] = '2.27540000e+04 3.73740000e+04 5.70000000e+01 3.28504878e+00 \
        7.46359145e+00 1.44378710e-04 2.08700000e+03 5.24300000e+03 \
            2.54110000e+04 3.35031203e+00 2.86000000e+02 3.95783263e-01 \
                3.65836802e-01 1.71500000e+04 2.0000000e+00 2.4000000e+01 \
                    7.11800000e+03 1.00720000e+04 2.22500000e+04'

    a[4] = '2.32230000e+04 3.83720000e+04 6.50000000e+01 3.30465487e+00 \
        7.12473505e+00 1.42307074e-04 2.13300000e+03 5.39800000e+03 \
            2.37180000e+04 3.06394523e+00 3.210000000e+02 2.91738889e-01 \
                3.64234230e-01 1.76130000e+04 2.0000000e+00 2.4000000e+01 \
                    7.22400000e+03 1.02480000e+04 2.28900000e+04'

    a[5] = '2.29610000e+04 3.85120000e+04 6.00000000e+01 3.35455773e+00 \
        7.15452555e+00 1.46104431e-04 2.1500000e+03 5.24500000e+03 \
            2.73430000e+04 3.57253604e+08 2.9500000e+02 3.77582632e-01 \
                3.65477216e-01 1.74720000e+04 2.000000e+00 2.500000e+01 \
                    7.01300000e+03 1.01410000e+04 2.18280000e+04'

    a[6] = '2.27400000e+04 3.82890000e+04 6.0000000e+01 3.36754617e+00 \
        7.11116298e+00 1.48095614e-04 2.0500000e+03 5.16800000e+03 \
            2.28940000e+04 3.02031662e+00 2.0500000e+02 2.74653887e-01 \
                3.67649799e-01 1.74040000e+04 2.0000000e+00 2.1000000e+01 \
                    7.0010000e+03 1.00190000e+04 2.23270000e+04'

    a[7] = '2.29680000e+04 3.96460000e+04 6.4000000e+01 3.45228144e+00 \
        7.09760298e+00 1.50314862e-04 2.01700000e+03 5.19600000e+03 \
            2.73860000e+04 3.57706374e+00 4.19000000e+02 3.45341840e-01 \
                3.65172906e-01 1.7870000e+04 2.000000e+00 3.0000000e+01 \
                    6.94800000e+03 1.02460000e+04 2.2500000e+04'
    for i in range(8):
        s = a[i].split()
        arr = np.array([float(i) for i in s])
        # print(i+6,arr)   
        np.save(f'../save2/paras_{i+6}.npy', arr)


def csv():
    out = []
    for i in range(22):
        para = np.load(f'../save2/paras_{i}.npy')
        out.append(para)

    out = np.array(out)
    print(out.shape)
    df = pd.DataFrame(out)
    df.columns = ['V', 'E', 'd_max', 'd_avg', 'l_avg', 'ρ', 'n_cc', 'n_c', \
            'T', 'T_avg', 'T_max', \
                'r', 'K_avg', \
                    'w_num', 'w_lb', 'w_hb',\
                        'n_b', 'n_lb', 'K_max']
    df.index = [str(year) for year in range(2000, 2000+22)]
    df.to_csv('../save2/analysis.csv')


if __name__ == '__main__':
    # main()
    # record()
    csv()