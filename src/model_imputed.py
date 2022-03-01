import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

class Impute():##Needs 2005N  2000~2004 P1P P1A P1V P1K
    def __init__(self,index_table_P1P,index_table_P1A,index_table_P1V,index_table_P1K,embeddings,embedding_size,train_year):
        self.index_table_P1P = index_table_P1P # list: [p, p]*5
        self.index_table_P1A = index_table_P1A # list: [p, a]*5
        self.index_table_P1V = index_table_P1V
        self.index_table_P1K = index_table_P1K
        self.embedding = embeddings # # list(array([n, 128])*5)
        self.embedding_size = embedding_size
        self.train_year = train_year
        self.realtion_weight = {'P1P_weight':tf.get_variable(name='P1P_weight',dtype=tf.float32,shape=()),
                                 'P1A_weight': tf.get_variable(name='P1A_weight', dtype=tf.float32,shape=()),
                                 'P1V_weight':tf.get_variable(name='P1V_weight',dtype=tf.float32,shape=()),
                                 'P1K_weight':tf.get_variable(name='P1K_weight',dtype=tf.float32,shape=())}
        self.outout_weights = []

    def rm_less0(self, x, val=0):
        x = x+1
        return x
    
    def pad(self, x):
        y = tf.zeros_like(x[0:1, :])
        return tf.concat([y, x], axis=0)

    def build(self):
        imputed_embeddings = []
        for i in range(self.train_year):
            # embedding[i]: [p, d], index_table_P1P[i]: [p, 100], 
            # print(self.embedding[i].shape, self.index_table_P1P[i].shape)
            # mm = int(self.embedding[i].shape[0])
            mm = 0
            embeds = self.pad(self.embedding[i])

            # embeds: [p, d], index_table_P1P[i]: [p, 100]
            lookup_ids = self.rm_less0(self.index_table_P1P[i], mm)
            output_P1P = tf.nn.embedding_lookup(embeds, ids=lookup_ids) # [p, 100, d]
            output_mean_P1P_sum = tf.reduce_sum(output_P1P, 1) # [p, d]
            idx_P1P = tf.to_float(tf.math.greater_equal(self.index_table_P1P[i], 0)) # [p, 100]
            idx_P1P = tf.reduce_sum(idx_P1P, 1, keepdims=True) # [p, 1]
            output_mean_P1P = output_mean_P1P_sum / (idx_P1P+0.001) # [p, d]
            self.output_mean_P1P = tf.multiply(output_mean_P1P, self.realtion_weight['P1P_weight']) # [p, d]

            lookup_ids = self.rm_less0(self.index_table_P1A[i], mm)
            output_P1A = tf.nn.embedding_lookup(embeds, ids=lookup_ids)
            output_mean_P1A_sum = tf.reduce_sum(output_P1A, 1)
            idx_P1A = tf.to_float(tf.math.greater_equal(self.index_table_P1A[i], 0))
            idx_P1A = tf.reduce_sum(idx_P1A, 1, keepdims=True)
            output_mean_P1A = output_mean_P1A_sum / (idx_P1A+0.001)
            self.output_mean_P1A = tf.multiply(output_mean_P1A, self.realtion_weight['P1A_weight'])

            lookup_ids = self.rm_less0(self.index_table_P1V[i], mm)
            output_P1V = tf.nn.embedding_lookup(embeds, ids=lookup_ids)
            output_mean_P1V_sum = tf.reduce_sum(output_P1V, 1)
            idx_P1V = tf.to_float(tf.math.greater_equal(self.index_table_P1V[i], 0))
            idx_P1V = tf.reduce_sum(idx_P1V, 1, keepdims=True)
            output_mean_P1V = output_mean_P1V_sum / (idx_P1V+0.001)
            self.output_mean_P1V = tf.multiply(output_mean_P1V, self.realtion_weight['P1V_weight'])

            lookup_ids = self.rm_less0(self.index_table_P1K[i], mm)
            output_P1K = tf.nn.embedding_lookup(embeds, ids=lookup_ids)
            output_mean_P1K_sum = tf.reduce_sum(output_P1K, 1)
            idx_P1K = tf.to_float(tf.math.greater_equal(self.index_table_P1K[i], 0))
            idx_P1K = tf.reduce_sum(idx_P1K, 1, keepdims=True)
            output_mean_P1K = output_mean_P1K_sum / (idx_P1K+0.001)
            self.output_mean_P1K = tf.multiply(output_mean_P1K, self.realtion_weight['P1K_weight'])

            output_mean = [output_mean_P1P,output_mean_P1A,output_mean_P1V,output_mean_P1K] # [4, p, d]
            imputed_embeddings.append(tf.reduce_mean(output_mean, 0)) # [p, d]
        imputed_embeddings = tf.concat(imputed_embeddings, axis=1) # [p, d*5]
        imputed_embeddings = tf.reshape(imputed_embeddings,shape=(-1,self.train_year,self.embedding_size)) # [p, 5, d]
        self.outout_weights = [self.realtion_weight['P1P_weight'],self.realtion_weight['P1A_weight'],
                               self.realtion_weight['P1V_weight'],self.realtion_weight['P1K_weight']]

        return imputed_embeddings,self.outout_weights
    

    def build_old(self):
        imputed_embeddings = []
        for i in range(self.train_year):
            # embedding[i]: [p, d], index_table_P1P[i]: [p, 100], 
            output_P1P = tf.nn.embedding_lookup(self.embedding[i], ids=self.index_table_P1P[i]) # [p, 100, d]
            output_mean_P1P_sum = tf.reduce_sum(output_P1P, 1) # [p, d]
            idx_P1P = tf.to_float(tf.math.greater_equal(self.index_table_P1P[i], 0)) # [p, d]
            idx_P1P = tf.reduce_sum(idx_P1P, 1, keepdims=True) # [p, 1]
            output_mean_P1P = output_mean_P1P_sum / (idx_P1P+0.001) # [p, d]
            self.output_mean_P1P = tf.multiply(output_mean_P1P, self.realtion_weight['P1P_weight']) # [p, d]

            output_P1A = tf.nn.embedding_lookup(self.embedding[i], ids=self.index_table_P1A[i])
            output_mean_P1A_sum = tf.reduce_sum(output_P1A, 1)
            idx_P1A = tf.to_float(tf.math.greater_equal(self.index_table_P1A[i], 0))
            idx_P1A = tf.reduce_sum(idx_P1A, 1, keepdims=True)
            output_mean_P1A = output_mean_P1A_sum / (idx_P1A+0.001)
            self.output_mean_P1A = tf.multiply(output_mean_P1A, self.realtion_weight['P1A_weight'])

            output_P1V = tf.nn.embedding_lookup(self.embedding[i], ids=self.index_table_P1V[i])
            output_mean_P1V_sum = tf.reduce_sum(output_P1V, 1)
            idx_P1V = tf.to_float(tf.math.greater_equal(self.index_table_P1V[i], 0))
            idx_P1V = tf.reduce_sum(idx_P1V, 1, keepdims=True)
            output_mean_P1V = output_mean_P1V_sum / (idx_P1V+0.001)
            self.output_mean_P1V = tf.multiply(output_mean_P1V, self.realtion_weight['P1V_weight'])

            output_P1K = tf.nn.embedding_lookup(self.embedding[i], ids=self.index_table_P1K[i])
            output_mean_P1K_sum = tf.reduce_sum(output_P1K, 1)
            idx_P1K = tf.to_float(tf.math.greater_equal(self.index_table_P1K[i], 0))
            idx_P1K = tf.reduce_sum(idx_P1K, 1, keepdims=True)
            output_mean_P1K = output_mean_P1K_sum / (idx_P1K+0.001)
            self.output_mean_P1K = tf.multiply(output_mean_P1K, self.realtion_weight['P1K_weight'])

            output_mean = [output_mean_P1P,output_mean_P1A,output_mean_P1V,output_mean_P1K] # [4, p, d]
            imputed_embeddings.append(tf.reduce_mean(output_mean, 0)) # [p, d]
        imputed_embeddings = tf.concat(imputed_embeddings, axis=1) # [p, d*5]
        imputed_embeddings = tf.reshape(imputed_embeddings,shape=(-1,self.train_year,self.embedding_size)) # [p, 5, d]
        self.outout_weights = [self.realtion_weight['P1P_weight'],self.realtion_weight['P1A_weight'],
                               self.realtion_weight['P1V_weight'],self.realtion_weight['P1K_weight']]

        return imputed_embeddings,self.outout_weights
