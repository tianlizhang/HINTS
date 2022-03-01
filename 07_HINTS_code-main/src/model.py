import tensorflow as tf
from model_embedding import RGCN_embedding
from model_imputed import Impute
from model_ts import CVAE
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from seq2seq import seq2seq

class Model():
    def __init__(self,adj_lsit,feature_list,embedding_size,input_seq,output_seq,
                 index_table_P1P,index_table_P1A,index_table_P1V,index_table_P1K,\
                     alignment_list,lr,batch_size,beta,\
                         gnn, method):

        # adj_list = [adj_1,...,adj_5], [adj_1_P1P, adj_1_P1A, adj_1_P1V, adj_1_P1K, adj_1_self]
        # feature = [feature_1, feature_2, feature_3, feature_4, feature_5]
        self.adj_list = adj_lsit 
        self.feature_list = feature_list
        self.index_table_P1P = index_table_P1P
        self.index_table_P1A = index_table_P1A
        self.index_table_P1V = index_table_P1V
        self.index_table_P1K = index_table_P1K
        self.alignment_list = alignment_list 
        self.train_year = 5
        self.embedding_size = embedding_size
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.lr = lr
        self.beta = beta
        self.batch_size = batch_size
        self.cvae = None
        self.impute = None
        self.embeddings = None
        self.imputed_embeddings = None
        self.loss = None
        self.align_loss = None
        self.outout_weights = None

        self.gnn = gnn # ['rgcn', gcn]
        self.method = method# ['cvae', 'seq2seq']
        print('Method:', self.method)

    def build(self):
        rgcn = RGCN_embedding(self.adj_list,self.feature_list,self.alignment_list,self.train_year,\
            self.gnn)
        self.embeddings,self.align_loss,self.W_1,self.W_2 = rgcn.build() # list: [p, d=128]*5
        self.impute = Impute(self.index_table_P1P,self.index_table_P1A,self.index_table_P1V,
              self.index_table_P1K,self.embeddings,self.embedding_size,self.train_year)
        self.imputed_embeddings,self.outout_weights = self.impute.build() # [p, 5, d]

        if self.method == 'seq2seq':
            self.cvae = seq2seq(time_length=self.train_year, imputed_size=self.embedding_size,
                        imputed_embeds=self.imputed_embeddings, input_seq=self.input_seq,
                        output_seq=self.output_seq, batch_size = self.batch_size)
        else:
            self.cvae = CVAE(time_length=self.train_year, imputed_size=self.embedding_size,
                            imputed_embeds=self.imputed_embeddings, input_seq=self.input_seq,
                            output_seq=self.output_seq, batch_size = self.batch_size)


        self.cvae.build_graph()

        self.loss = tf.reduce_mean(self.cvae.pred_loss) + self.align_loss*self.beta
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
