'''In this script, we construct three simple classifiers accounting for three
tasks: . All of them are based on a RGCN backbone.
'''
import argparse
import os
import time

import dgl
from dgl.dataloading import negative_sampler
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
import dgl.function as fn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, average_precision_score
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from batch_loader import TemporalEdgeCollator, TemporalEdgeDataLoader
from batch_model import BatchModel
from model import Model, compute_loss, construct_negative_graph
from util import (CSRA_PATH, DBLP_PATH, NOTE_PATH, EarlyStopMonitor,
                  set_logger, set_random_seed, write_result)


def load_data():
    coauthors = []
    for year in range(2000, 2004):
        g_list, _ = dgl.load_graphs(f'../save/coauthor_{year}.graph')
        g = g_list[0]
        n_nodes = len(g.nodes())
        g.ndata['feat'] = torch.rand(n_nodes, 300)
        coauthors.append(g)
    return coauthors


def calc_pos_weight(coauthors, train_range):
    total, l = 0, []
    for i in train_range:
        g = coauthors[i]
        topic = g.edata['topic']
        total += len(topic)
        topic_posnum = topic.sum(0)
        l.append(topic_posnum)
    l = torch.vstack(l)
    pos_num = l.sum(0)
    pos_weight = (total-pos_num)/pos_num
    return pos_weight


def main(args):
    coauthors, _ = dgl.load_graphs(f'../save2/coauthors_topic.graph') #load_data()
    # print(f'coauthors:{coauthors}')
    device = torch.device('cuda:{}'.format(args.gpu))
    # coauthors = [g.to(device) for g in coauthors]
    # node_features = coauthors[0].edata['feat']
    n_features = 300#node_features.shape[1]
    k = 5
    model = BatchModel(n_features, 100, 100).to(device)
    opt = torch.optim.Adam(model.parameters())

    train_idx = int(len(coauthors) * 0.75)
    # features = [g.ndata['feat'] for g in coauthors]
    features = [torch.rand(len(g.nodes()), 300) for g in coauthors]
    num_nodes = coauthors[0].number_of_nodes()
    num_edges = sum([g.number_of_edges() for g in coauthors])

    sampler = MultiLayerNeighborSampler([15, 10])
    neg_sampler = None #negative_sampler.Uniform(5)
    train_range = list(range(1, train_idx))
    test_range = list(range(train_idx, len(coauthors)))
    train_loader = TemporalEdgeDataLoader(coauthors, train_range, 
        sampler, negative_sampler=neg_sampler, batch_size=1024, shuffle=False,
        drop_last=False, num_workers=0)
    test_loader = TemporalEdgeDataLoader(coauthors, test_range,
        sampler, negative_sampler=neg_sampler, batch_size=1024, shuffle=False,
        drop_last=False, num_workers=0)

    # criterion = nn.L1Loss(reduce='mean')
    pos_weight = calc_pos_weight(coauthors, train_range)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    logger.info('Begin training with %d nodes, %d edges.', num_nodes, num_edges)
    for epoch in range(100):
        loss_avg = 0

        model.train()
        batch_bar = tqdm(train_loader)
        for step, (input_nodes, pos_graph, history_blocks) in enumerate(batch_bar):
            history_inputs = [nfeat[nodes].to(device) for nfeat, nodes in zip(features, input_nodes)]
            # batch_inputs = nfeats[input_nodes].to(device)
            pos_graph = pos_graph.to(device)
            # neg_graph = neg_graph.to(device)
            # blocks = [block.int().to(device) for block in blocks]
            history_blocks = [[block.int().to(device) for block in blocks] for blocks in history_blocks]

            pos_score = model(history_blocks, history_inputs, pos_graph)
            # loss = compute_loss(pos_score)
            loss = criterion(pos_score, pos_graph.edata['topic'])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_avg += loss.item()
            batch_bar.set_postfix(loss=round(loss.item(), 4))

        loss_avg /= len(train_loader)
        y_probs = []
        y_labels = []
        val_loss, total = 0, 0
        model.eval()
        with torch.no_grad():
            for step, (input_nodes, pos_graph, history_blocks) in enumerate(tqdm(test_loader)):
                history_inputs = [nfeat[nodes].to(device) for nfeat, nodes in zip(features, input_nodes)]
                # batch_inputs = nfeats[input_nodes].to(device)
                pos_graph = pos_graph.to(device)
                # neg_graph = neg_graph.to(device)
                # blocks = [block.int().to(device) for block in blocks]
                history_blocks = [[block.int().to(device) for block in blocks] for blocks in history_blocks]

                pos_score = model(history_blocks, history_inputs, pos_graph)
                label_score = pos_graph.edata['topic']
                val_loss += criterion(pos_score, label_score)
                # pos_score = pos_score.flatten()
                # label = label_score.flatten()
                # total += len(pos_score)
                y_probs.append(pos_score.detach().cpu().numpy())
                y_labels.append(label_score.cpu().numpy())
                # y_labels.append(np.ones_like(y_probs[-1]))
                # y_probs.append(neg_score.detach().cpu().numpy())
                # y_labels.append(np.zeros_like(y_probs[-1]))
        val_loss /= len(test_loader)
        
        # y_probs = [y.squeeze(1) for y in y_probs]
        # y_labels = [y.squeeze(1) for y in y_labels]
        # y_prob = np.hstack(y_probs)
        y_pred = np.vstack(y_probs) > 0.5
        y_label = np.vstack(y_labels)
        print(f"y_label:{y_label.shape}, y_pred:{y_pred.shape}")
        # ap = average_precision_score(y_label, y_prob)
        # auc = roc_auc_score(y_label, y_prob)
        f1 = f1_score(y_label, y_pred, average='micro')
        f1_mac = f1_score(y_label, y_pred, average='macro')
        print(f"Epoch:{epoch}, train_loss:{loss_avg}, val_loss:{val_loss}, f1:{f1}, f1_mac:{f1_mac}")
        # logger.info('Epoch %03d Training loss: %.4f, Test F1: %.4f, AP: %.4f, AUC: %.4f', epoch, loss_avg, f1, ap, auc)


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
