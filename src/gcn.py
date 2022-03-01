import tensorflow as tf
# from tensorflow.keras import activations, regularizers, constraints, initializers
# spdot = tf.sparse.sparse_dense_matmul
# dot = tf.matmul
import numpy as np

class GCN():
    def __init__(self,adj,feature,out_dims,graph_order,dropout=False):
        self.adj = adj
        self.feature = feature # [n, 4]
        self.input_dim = self.feature.shape[1] # 4
        self.out_dims = out_dims
        self.relation = ['P1P','P1A','P1V','P1K','self']
        self.support = len(self.relation)
        self.dropout = dropout
        self.graph_order = graph_order
        self.W_1 = None
        self.W_2 = None
        self.activation = tf.nn.relu

        self.adj_all = []
        self.build()

    def add_weight(self, relation, layer_number, input_dim, out_dim):
        return tf.get_variable(shape = (input_dim,out_dim), \
            name='graph_order_{}_{}_W_{}'.format(self.graph_order, relation, layer_number))

    def build(self):
        features_shape = tf.shape(self.feature)
        self.W_1 = self.add_weight(relation='W_1', layer_number=1,
                            input_dim=self.input_dim, out_dim=self.out_dims["out_dim1"]) # [4, 64]
        self.W_2 = self.add_weight(relation='W_2',layer_number=2,
                        input_dim=self.out_dims["out_dim1"],
                        out_dim=self.out_dims["out_dim2"]) # [64, 128]


    def call(self): 
        if self.adj_all==[]:
            adj_sum = self.adj[0]
            for i in range(1, self.support):
                adj_sum = tf.sparse.add(adj_sum, self.adj[i])
            adj_all = tf.cast(adj_sum, dtype=tf.bool)
            self.adj_all = tf.cast(adj_all, dtype=tf.float32)
        
        supports = tf.sparse.sparse_dense_matmul(self.adj_all, self.feature) # [n, 4]
        H_1 = tf.matmul(supports,self.W_1) # [n, 64]
        H_1 = self.activation(H_1)
        H_1 = tf.nn.dropout(H_1, 0.2)

        supports = tf.sparse.sparse_dense_matmul(self.adj_all, H_1)
        H_2 = tf.matmul(supports,self.W_2)
        return  H_2


# class GCNConv(tf.keras.layers.Layer):
#     def __init__(self,
#                 units,
#                 activation=lambda x: x,
#                 use_bias=True,
#                 kernel_initializer='glorot_uniform',
#                 kernel_regularizer=None,
#                 kernel_constraint=None,
#                 bias_initializer='zeros',
#                 bias_regularizer=None,
#                 bias_constraint=None,
#                 activity_regularizer=None,
#                 **kwargs):
#         # 初始化不需要训练的参数
#         self.units = units
#         # activation=None 使用线性激活函数（等价不使用激活函数）
#         self.activation = activations.get(activation)
#         self.use_bias = use_bias
        
#         # 初始化方法定义了对Keras层设置初始化权重（bias）的方法 glorot_uniform
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
        
#         # 加载正则化的方法
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.activity_regularizer = regularizers.get(activity_regularizer)
        
#         # 约束：对权重值施加约束的函数。
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)

#         super(GCNConv, self).__init__()

#     def build(self, input_shape):
#         """ GCN has two inputs : [shape(An), shape(X)]
#         """
#         # gsize = input_shape[0][0]  # graph size
#         fdim = input_shape[1][1]  # feature dim
        
#         # hasattr 检查该对象self是否有某个属性'weight'
#         if not hasattr(self, 'weight'):
#             self.weight = self.add_weight(name="weight",
#                                         shape=(fdim, self.units),
#                                         initializer=self.kernel_initializer,
#                                         constraint=self.kernel_constraint,
#                                         trainable=True)
#         if self.use_bias:
#             if not hasattr(self, 'bias'):
#                 self.bias = self.add_weight(name="bias",
#                                             shape=(self.units, ),
#                                             initializer=self.bias_initializer,
#                                             constraint=self.bias_constraint,
#                                             trainable=True)
#         super(GCNConv, self).build(input_shape)

    
#     def call(self, inputs):
#         """ GCN has two inputs : [An, X]
#             对称归一化版本的GCN的核心公式计算过程
#         """
#         self.An = inputs[0]
#         self.X = inputs[1]
#         # if isinstance(self.X, tf.SparseTensor):
#         h = tf.sparse.sparse_dense_matmul(self.X, self.weight)
#         # else:
#         #     # 二维数组矩阵之间的dot函数运算得到的乘积是矩阵乘积
#         #     h = dot(self.X, self.weight) # h = X*W
#         output = tf.sparse.sparse_dense_matmul(self.An, h) # o = A*h

#         if self.use_bias:
#             output = tf.nn.bias_add(output, self.bias)

#         if self.activation:
#             output = self.activation(output)

#         return output
    

    
