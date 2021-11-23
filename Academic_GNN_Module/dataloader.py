from dgl.dataloading import GraphDataLoader
from networkx.readwrite.multiline_adjlist import parse_multiline_adjlist
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import math
import numpy as np
import dgl
import torch

def data_loader(args, coauthors, features, train_ratio=0.75, test_ratio=0.25, collate_fn=None):
    dataset = []
    num = len(coauthors)
    for idx in range(3, len(coauthors)):
        # dataset.append((coauthors[:idx], features[:idx], coauthors[idx]))
        dataset.append((coauthors[idx-3:idx], features[idx-3:idx], coauthors[idx], np.array([idx])))
        # dataset.append((coauthors[idx-3:idx], features[idx-3:idx], coauthors[idx], np.array([idx])))
        # graph_data = {
        #     ('author', 'write', 'paper'): ([0, 1], [1,2])
        # }
        # g_pad, x_pad = dgl.heterograph(graph_data), torch.zeros(3, 300)
        # his_gs = coauthors[:idx]
        # his_xs = features[:idx]
        # for j in range(idx, num):
        #     his_gs.append(g_pad.to(args.device))
        #     his_xs.append(x_pad.to(args.device))
        # dataset.append((his_gs, his_xs, coauthors[idx], np.array([idx])))
        # dataset.append((coauthors, features, coauthors[idx], np.array([idx])))
    num = len(dataset)
    indices = list(range(num))
    cut1 = math.floor(train_ratio * num)
    cut2 = math.floor((1-test_ratio) * num)
    train_idx, valid_idx, test_idx = indices[:cut1], indices[cut1:cut2], indices[cut2:]
    # print(len(dataset))
    # print(train_idx, valid_idx, test_idx)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = GraphDataLoader(dataset, sampler=train_sampler,
            batch_size=args.batch_size, collate_fn=collate_fn)
    valid_loader = GraphDataLoader(dataset, sampler=valid_sampler,
        batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = GraphDataLoader(dataset, sampler=test_sampler,
        batch_size=args.batch_size, collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader
        


class GINDataLoader():
    def __init__(self, dataset, batch_size, device, collate_fn=None, seed=0,shuffle=True, \
                split_name='fold10', fold_idx=0, train_proportion=0.7, test_proportion=0.1):
        self.shuffle = shuffle
        self.seed = seed
        self.kwargs = {'pin_memory': True} if 'cuda' in device.type else {}
        labels = [l for _, l,_ in dataset]

        if split_name == 'fold10':
            train_idx, valid_idx = self._split_fold10(labels, fold_idx, seed, shuffle)
        elif split_name == 'rand':
            train_idx, valid_idx = self._split_rand(labels)
        else:
            raise NotImplementedError()
        num_entries = len(labels)
        indices = list(range(num_entries))
        train_end = int(math.floor(train_proportion * num_entries))
        test_start = int(math.floor((1-test_proportion) * num_entries))
        train_idx, valid_idx, test_idx = indices[:train_end], indices[train_end:test_start], indices[test_start:]
        print("graphs_num:{}, train_data:{}, valid_data:{}, test_data:{}".format\
                    (num_entries,len(train_idx),len(valid_idx),len(test_idx)))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        print("train_sampler : valid_sampler = {} : {}".format(
           len(train_sampler), len(valid_sampler)))

        self.train_loader = GraphDataLoader(dataset, sampler=train_sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs)
        self.valid_loader = GraphDataLoader(dataset, sampler=valid_sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs)
        self.test_loader = GraphDataLoader(dataset, sampler=test_sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs)
        print("train_loader : valid_loader = {} : {}".format(
           len(self.train_loader), len(self.valid_loader)))

    def train_valid_test_loader(self):
        return self.train_loader, self.valid_loader, self.test_loader

    def train_valid_loader(self):
        return self.train_loader, self.valid_loader

    def _split_fold10(self, labels, fold_idx=0, seed=0, shuffle=True):
        assert 0 <= fold_idx and fold_idx < 10, print("fold_idx must be from 0 to 9.")

        skf = StratifiedKFold(n_splits=10, shuffle=shuffle, random_state=seed)
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):    # split(x, y)
            idx_list.append(idx)
        train_idx, valid_idx = idx_list[fold_idx]
        print("train_set : test_set = %d : %d", len(train_idx), len(valid_idx))
        return train_idx, valid_idx

    def _split_rand(self, labels, split_ratio=0.7, seed=0, shuffle=True):
        num_entries = len(labels)
        indices = list(range(num_entries))
        np.random.seed(seed)
        np.random.shuffle(indices)
        split = int(math.floor(split_ratio * num_entries))
        train_idx, valid_idx = indices[:split], indices[split:]

        print("train_set : test_set = %d : %d",len(train_idx), len(valid_idx))
        return train_idx, valid_idx