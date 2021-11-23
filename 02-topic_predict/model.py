"""RGCN layer implementation"""
from collections import defaultdict

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
import tqdm
import torch


class DotProductPredictor(nn.Module):
    def __init__(self, in_dim, out_dim=1000, hidden_dim=100):
        super().__init__()
        self.W1 = nn.Linear(in_dim*2, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_dim)
        self.criterion = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()

    def dot_linear_func(self, edges):
        # h = torch.dot(edges.src['h'], edges.dst['h'])
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h)).squeeze(1))}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.dot_linear_func)
            return graph.edata['score']
            # return self.criterion(graph.edata['score'], graph.edata['onehot'])


def construct_negative_graph(graph, k):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k, )).to(neg_src)
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation,
             dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, inputs):
        inputs = inputs.reshape(-1, 300)
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(
                g.num_nodes(),
                self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()).to(g.device),
                sampler,
                device=device if num_workers == 0 else None,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features,
        2, nn.ReLU(), 0.2)
        self.lstm = nn.LSTM(out_features, out_features, 1)
        self.pred = DotProductPredictor()
        self.out_features = out_features

    def forward(self, history_gs, xs, g, neg_g):
        hs = [self.sage(g, x) for g, x in zip(history_gs, xs)]
        hs = torch.stack(hs) # (seq, batch, dim)
        hs, _ = self.lstm(hs) # default (h0, c0) are all zeros
        h = hs[-1]
        print(hs.shape, h.shape)
        return self.pred(g, h), self.pred(neg_g, h)
    
    def forward2(self, history_gs, xs, g, neg_g, idx):
        # idx：[b, 1]
        max_idx = torch.max(idx)
        hs = []
        for i, (g, x) in enumerate(zip(history_gs, xs)):
            if i<max_idx:
                h = self.sage(g, x)
            
        return 0



def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean()
