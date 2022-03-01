import numpy as np
import tensorflow as tf
from rgcn import RGCN
from gcn import GCN
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


class RGCN_embedding():
    def __init__(self, adj_list,feature_list,alignment_index,train_year,\
        gnn):
        self.adj_list = adj_list
        self.feature_list = feature_list
        # adj_list = [adj_1,...,adj_5], adj_1 = [adj_1_P1P, adj_1_P1A, adj_1_P1V, adj_1_P1K, adj_1_self]
        # [feature_1, feature_2, feature_3, feature_4, feature_5]
        # alignment_list = [alignment_list_1,alignment_list_2,alignment_list_3,alignment_list_4]
            # list, len=10, 2000-2009
        self.alignment_index = alignment_index  
        self.train_year = train_year
        self.out_dims = {
            "out_dim1":64,
            "out_dim2":128,
        }
        self.rgcn_object = None

        self.gnn = gnn # ['rgcn', 'gcn']
        print(f'Gnn embedding method: {self.gnn}')

    def build(self):
        embeddings = []
        for i in range(self.train_year):
            if self.gnn == 'gcn':
                self.rgcn_object = GCN(self.adj_list[i],self.feature_list[i],self.out_dims,i)
            else:
                self.rgcn_object = RGCN(self.adj_list[i],self.feature_list[i],self.out_dims,i)
            embeddings.append(self.rgcn_object.call()) # list(array([n, 128])*5)


        align_loss = 0 # embeddings: list(array([n, 128])*5years)
        for i in range(self.train_year-1): # alignment_index: list([ind1, ind2]*5years)
            align_embeds1 = tf.nn.embedding_lookup(embeddings[i], ids=self.alignment_index[i][0])
            align_embeds2 = tf.nn.embedding_lookup(embeddings[i+1], ids=self.alignment_index[i][1])
            align_loss = align_loss + \
                (1/(self.train_year-1))*tf.norm((align_embeds1 - align_embeds2), ord='euclidean') \
                    / tf.to_float(tf.shape(align_embeds1)[0])

        return  embeddings,align_loss,self.rgcn_object.W_1,self.rgcn_object.W_2,


