'''In this script, we construct three simple classifiers accounting for three
tasks: . All of them are based on a RGCN backbone.
'''
import argparse
import os
import time

import dgl
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
    
    logger.info('Split co-author into each year.')
    coauthors = []
    for year in range(2000, 2022):
        write_eids = g.filter_edges(lambda x: x.data['year'] == year, etype='write')
        written_eids = g.filter_edges(lambda x: x.data['year'] == year, etype='written')

        year_graph = g.edge_subgraph({'write':write_eids, 'written': written_eids}, preserve_nodes=True)
        year_graph.update_all(fn.copy_src('title', 'h'), fn.mean('h', 'feat'), etype=('paper', 'written', 'author'))
        coauthor = dgl.metapath_reachable_graph(year_graph, ['write', 'written'])
        # print('year:{}, authors:{}, coauthor-links:{}'.format(year, coauthor.num_nodes(), coauthor.num_edges()))
        coauthors.append(coauthor)
    return coauthors


def main(args):
    graph = load_data()
    coauthors = build_coauthor_graph(graph, years=range(2000, 2022))
    device = torch.device('cuda:{}'.format(args.gpu))
    coauthors = [g.to(device) for g in coauthors]
    node_features = coauthors[0].ndata['feat']
    n_features = node_features.shape[1]
    k = 5
    model = Model(n_features, 100, 100).to(device)
    opt = torch.optim.Adam(model.parameters())

    train_idx = int(len(coauthors) * 0.75)
    features = [g.ndata['feat'] for g in coauthors]
    train_graphs = coauthors[:train_idx]
    test_graphs = coauthors[train_idx:]
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
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")

    logger = set_logger()
    set_random_seed(seed=42)
    args = parser.parse_args()
    logger.info(args)
    main(args)
