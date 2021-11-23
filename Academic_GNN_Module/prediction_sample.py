'''In this script, we construct three simple classifiers accounting for three
tasks: . All of them are based on a RGCN backbone.
'''
import argparse
import os
import time

import dgl
from dgl.dataloading import negative_sampler
import dgl.function as fn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, average_precision_score
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model import Model, compute_loss, construct_negative_graph
from util import (CSRA_PATH, DBLP_PATH, NOTE_PATH, EarlyStopMonitor,
                  set_logger, set_random_seed, write_result)


def load_data():
    # logger.info('Load authors, papers, edges.')
    # authors = pd.read_csv(f'{DBLP_PATH}/dblp.authors.fillorg.csv')
    # papers = pd.read_csv(f'{DBLP_PATH}/dblp.papers.csv')
    # edges = pd.read_csv(f'{DBLP_PATH}/dblp.edges.csv')

    logger.info('Load paper title embeddings.')
    title_path = 'cache/dblp.papers.title.npy'
    title_path = NOTE_PATH + title_path
    title_embeds = np.load(title_path)
    logger.info('Title embeds: {}'.format(title_embeds.shape))

    logger.info('Load a heterogeneous graph with (author, paper, org, country, topic).')
    graph_path = f'{DBLP_PATH}/dblp.graph'
    graph_list, label_list = dgl.load_graphs(graph_path)
    graph = graph_list[0]
    graph.nodes['paper'].data['title'] = torch.from_numpy(title_embeds).float()
#     init_types = ['author', 'org', 'country', 'topic']
#     for node in init_types:
#         shape = graph.number_of_nodes(node), title_embeds.shape[1]
#         w = torch.empty(*shape)
#         w.requires_grad = True
#         torch.nn.init.xavier_uniform_(w)
#         graph.nodes[node].data['init'] = w
    logger.info('Graph %s.', str(graph))
    return graph

def build_coauthor_graph(graph, years=None, freq=50):
    if years is None:
        logger.info('Default year range is 2000-2021(included).')
        years = range(2000, 2022)

    graph = graph.node_type_subgraph(['author', 'paper'])
    num_paper = graph.number_of_nodes('paper')
    logger.info('Filter author less than %d papers.', freq)
    author_ids = torch.where(graph.out_degrees(etype='write') >= freq)[0]
    paper_ids = torch.arange(num_paper)
    g = graph.subgraph({'author': author_ids, 'paper': paper_ids})
    # print("g:", g)
    # print("g.edata", g.edata)
    # print("g.ndata", g.ndata)
    # return [g]
    logger.info('Split co-author into each year.')
    coauthors = []
    for year in range(2021, 2022):
        write_eids = g.filter_edges(lambda x: x.data['year'] < 2055, etype='write') # x is a edge in g
        written_eids = g.filter_edges(lambda x: x.data['year'] < 2055, etype='written')

        year_graph = g.edge_subgraph({'write':write_eids, 'written': written_eids}, preserve_nodes=True)
        year_graph.update_all(fn.copy_src('title', 'h'), fn.mean('h', 'feat'), etype=('paper', 'written', 'author'))
        coauthor = dgl.metapath_reachable_graph(year_graph, ['write', 'written'])
        # print('year:{}, authors:{}, coauthor-links:{}'.format(year, coauthor.num_nodes(), coauthor.num_edges()))
        coauthors.append(coauthor)
        g = coauthor
        print("g:", g)
        print("g.edata", g.edata)
        print("g.ndata", g.ndata)
    return coauthors

# Paper = namedtuple('Paper', ['pid', 'year', 'title', 'keywords'])
# Author = namedtuple('Author', ['aid', 'name', 'year', 'org'])
# Edge = namedtuple('Edge', ['pid', 'aid', 'year', 'title'])
class myNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts=[5]):
        super().__init__(len(fanouts))
        self.fanouts = fanouts

    def sample_frontier(self, block_id, g, seed_nodes):
        g = dgl.node_subgraph()
        g = dgl.in_subgraph(g, seed_nodes)
        print("01-g:", g)
        print("02-seed_nodes", seed_nodes)
        years =  g.edata['year'][('author', 'write', 'paper')]
        # {('author', 'write', 'paper'): tensor([]), ('paper', 'written', 'author'): tensor([])}
        print("03-years", years)

        for year in years:
            print('04-year:', year)
            write_eids = g.filter_edges(lambda x: x.data['year'] < year, etype='write') # x is a edge in g
            print("year:{}, write_eids:{}", year, write_eids)
        print("ndata:", g.ndata)
        print("edata:", g.edata)
        # print(ztl)

        dgl.node_subgraph()
        sub_g = dgl.in_subgraph(g, seed_nodes)
        print("03-sub_g:",sub_g)
        print("04-ndata['year']", sub_g.ndata['year'])
        print("05-edata[year]", sub_g.edata['year'])
        write_eids = sub_g.filter_edges(lambda x: x.data['year'] <= sub_g.ndata['year'], etype='write')
        print(write_eids)
        # print(ztl)
        sub_g.apply_edges(fn.copy_src('year', 'h'))
        write_eids = sub_g.filter_edges(lambda x: x.data['h'] <= sub_g.ndata['year'], etype='write')
        

        fanout = self.fanouts[block_id]
        if fanout is None:
            frontier = dgl.in_subgraph(g, seed_nodes)
        else:
            frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
        return frontier


def main(args):
    graph = load_data()
    coauthors = build_coauthor_graph(graph, years=range(2000, 2022))
    device = torch.device('cuda:{}'.format(args.gpu))
    k = 5
    # sampler = dgl.dataloading.MultiLayerNeighborSampler(
    #     [int(fanout) for fanout in args.fan_out.split(',')])
    # sampler = myNeighborSampler()
    # neg_sampler = dgl.dataloading.negative_sampler.Uniform(k)
    # dataloader = dgl.dataloading.EdgeDataLoader(
    #     coauthors[-1], torch.arange(coauthors[-1].number_of_edges()), 
    #     sampler,
    #     # negative_sampler=NegativeSampler(g, args.num_negs, args.neg_share),
    #     negative_sampler=neg_sampler,
    #     device=device,
    #     # use_ddp=n_gpus > 1,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=1)

    sampler = myNeighborSampler()
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(k)
    # train_eid = torch.arange(coauthors[-1].number_of_edges())
    train_eid = torch.arange(100)
    dataloader = dgl.dataloading.EdgeDataLoader(
        coauthors[-1], 
        {'write': train_eid}, 
        sampler,
        # negative_sampler=NegativeSampler(g, args.num_negs, args.neg_share),
        exclude='reverse_types',
        reverse_etypes={'write': 'written', 'written': 'write'},
        negative_sampler=neg_sampler,
        device=device,
        # use_ddp=n_gpus > 1,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1)
    input_nodes, pos_graph, neg_graph, mfgs = next(iter(dataloader))

    coauthors = [g.to(device) for g in coauthors]
    node_features = coauthors[0].ndata['feat']
    n_features = node_features.shape[1]
    
    model = Model(n_features, 100, 100).to(device)
    opt = torch.optim.Adam(model.parameters())

    train_idx = int(len(coauthors) * 0.75)
    features = [g.ndata['feat'] for g in coauthors]
    # train_graphs = coauthors[:train_idx]
    # test_graphs = coauthors[train_idx:]
    num_nodes = coauthors[0].number_of_nodes()
    num_edges = sum([g.number_of_edges() for g in coauthors])

    logger.info('Begin training with %d nodes, %d edges.', num_nodes, num_edges)
    for epoch in range(100):
        loss_avg = 0
        for idx in range(1, train_idx):
            history_gs = coauthors[:idx]
            history_xs = features[:idx]
            graph = coauthors[idx]
            negative_graph = construct_negative_graph(graph, k)
            pos_score, neg_score = model(history_gs, history_xs, graph, negative_graph)
            loss = compute_loss(pos_score, neg_score)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print(loss.item())
            loss_avg += loss.item()
        loss_avg /= train_idx

        y_probs = []
        y_labels = []
        for idx in range(train_idx, len(coauthors)):
            history_gs = coauthors[:idx]
            history_xs = features[:idx]
            graph = coauthors[idx]
            negative_graph = construct_negative_graph(graph, k)
            pos_score, neg_score = model(history_gs, history_xs, graph, negative_graph)
            # print(pos_score.shape, neg_score.shape)
            y_probs.append(pos_score.detach().cpu().numpy())
            y_labels.append(np.ones_like(y_probs[-1]))
            y_probs.append(neg_score.detach().cpu().numpy())
            y_labels.append(np.zeros_like(y_probs[-1]))
        
        y_probs = [y.squeeze(1) for y in y_probs]
        y_labels = [y.squeeze(1) for y in y_labels]
        y_prob = np.hstack(y_probs)
        y_pred = np.hstack(y_probs) > 0.5
        y_label = np.hstack(y_labels)
        ap = average_precision_score(y_label, y_prob)
        auc = roc_auc_score(y_label, y_prob)
        f1 = f1_score(y_label, y_pred)
        logger.info('Epoch %03d Training loss: %.4f, Test F1: %.4f, AP: %.4f, AUC: %.4f', epoch, loss_avg, f1, ap, auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n-hidden", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-bases", type=int, default=-1)
    parser.add_argument("--n-layers", type=int, default=2)

    logger = set_logger()
    set_random_seed(seed=42)
    args = parser.parse_args()
    logger.info(args)
    main(args)
