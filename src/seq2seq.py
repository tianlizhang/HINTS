#-*- coding:utf-8 -*-
import numpy as np
from scipy.special import erf
import tensorflow as tf
# from tensorflow.contrib.seq2seq import *
# from tensorflow.python.layers.core import Dense


class seq2seq():
    def __init__(self,time_length,imputed_size,imputed_embeds,input_seq,output_seq,batch_size):
        self.imputed_size = imputed_size
        self.batch_size = batch_size
        self.time_length = time_length
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.imputed_embeds = imputed_embeds
        self.eta = None
        self.mu = None
        self.sigma = None
        self.pred_loss = None
        # self.dropout = tf.placeholder(tf.float32)

    def build_graph(self):
        mlp_hidden_unit = {"encoder_1":50,"encoder_2":10,
                           "decoder_1":20,"decoder_2":8,"decoder_3":1,
                           "rnn":50,"conditional_1":20}

        # encoder 过程
        with tf.variable_scope('encode', reuse=False):
            enc_cell = tf.nn.rnn_cell.BasicRNNCell(mlp_hidden_unit["rnn"]) 
            # enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=self.dropout) # 进行 dropout
            encoder_outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, self.imputed_embeds, dtype=tf.float32) 
            #  enc_states : [batch_size, hidden], encoder_outputs:[batch_size, s, hidden]

        # decoder 过程
        with tf.variable_scope('decode', reuse=False):
            dec_cell = tf.nn.rnn_cell.BasicRNNCell(mlp_hidden_unit["rnn"])
            # dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=self.dropout)  # 进行 dropout
            outputs, _ = tf.nn.dynamic_rnn(dec_cell, encoder_outputs, initial_state=enc_states, dtype=tf.float32) 
            # outputs : [batch_size, 5, 50]

            h1 = tf.layers.dense(outputs, mlp_hidden_unit["decoder_1"], activation=tf.nn.relu) # [b, s, 20]
            h2 = tf.layers.dense(h1, mlp_hidden_unit["decoder_2"], activation=tf.nn.relu) # [b, s, 8]
            h3 = tf.layers.dense(h2, mlp_hidden_unit["decoder_3"], activation=tf.nn.relu) # [b, s, 1]
            self.citation_pred = tf.squeeze(h3, axis=2)

        self.pred_loss = tf.reduce_sum(tf.square(self.citation_pred-self.output_seq),1)


        # encoder 过程
        with tf.variable_scope('encode', reuse=True):
            enc_cell = tf.nn.rnn_cell.BasicRNNCell(mlp_hidden_unit["rnn"]) 
            # enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=self.dropout) # 进行 dropout
            encoder_outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, self.imputed_embeds, dtype=tf.float32) 
            #  enc_states : [batch_size, hidden], encoder_outputs:[batch_size, s, hidden]

        # decoder 过程
        with tf.variable_scope('decode', reuse=True):
            dec_cell = tf.nn.rnn_cell.BasicRNNCell(mlp_hidden_unit["rnn"])
            # dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=self.dropout)  # 进行 dropout
            outputs, _ = tf.nn.dynamic_rnn(dec_cell, encoder_outputs, initial_state=enc_states, dtype=tf.float32) 
            # outputs : [batch_size, 5, 50]

            h1 = tf.layers.dense(outputs, mlp_hidden_unit["decoder_1"], activation=tf.nn.relu) # [b, s, 20]
            h2 = tf.layers.dense(h1, mlp_hidden_unit["decoder_2"], activation=tf.nn.relu) # [b, s, 8]
            h3 = tf.layers.dense(h2, mlp_hidden_unit["decoder_3"], activation=tf.nn.relu) # [b, s, 1]
            self.citation_pred_test = tf.squeeze(h3, axis=2)