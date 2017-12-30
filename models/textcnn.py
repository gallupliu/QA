# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: textcnn.py
@time: 2017/12/30 14:53


"""


import tensorflow as tf
from  models import model_utils
from  models.model import Model
from utils.utils import feature2cos_sim, max_pooling, cal_loss_and_acc


class TextCNN(Model):
    def __init__(self, config,embedding):
        # self.config = config
        # self.embeddings = embedding
        super(TextCNN, self).__init__(config,embedding)
        self.add_placeholder()
        query_ids, left_ids, right_ids = self.add_embedding()
        query_feature = self.build(query_ids)
        tf.get_variable_scope().reuse_variables()
        left_feature = self.build(left_ids)
        tf.get_variable_scope().reuse_variables()
        right_feature = self.build(right_ids)

        self.ori_cand = feature2cos_sim(query_feature,left_feature)
        self.ori_neg = feature2cos_sim(query_feature,right_feature)
        self.loss, self.acc = cal_loss_and_acc(self.ori_cand, self.ori_neg)
        self.train_op = self.train_op(self.loss)



    def add_placeholder(self):
        """
        :功能： define input variable
        :return:
        """
        self.keep_prob = tf.placeholder(tf.float32, name="keep_drop")

        self.lr = tf.Variable(0.0, trainable=False)
        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self.new_lr)

        self.query = tf.placeholder(tf.int32, shape=[None, self.config['inputs']['share']['text1_maxlen']],name="query")
        self.input_left = tf.placeholder(tf.int32, shape=[None, self.config['inputs']['share']['text1_maxlen']],name="left")
        self.input_right= tf.placeholder(tf.int32, shape=[None, self.config['inputs']['share']['text1_maxlen']],name="right")

    def add_embedding(self):
        """
        :功能：对输入建立索引
        :return:
        """
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            W = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
            query_ids = tf.nn.embedding_lookup(W, self.query)
            left_ids = tf.nn.embedding_lookup(W, self.input_left)
            right_ids = tf.nn.embedding_lookup(W, self.input_right)
        return query_ids,left_ids,right_ids

    def build(self,ids):
        """
        :param ids:
        :return:
        """
        with tf.variable_scope("TEXTCNN_scope", reuse=None):
            feature = model_utils.text_cnn(ids,filter_sizes,num_filters,embedding_size,sequence_length,dropout_keep_prob=1.0)
        return feature

    def train_op(self,loss):
        """
        :param loss:
        :return:
        """
        self.global_step = tf.Variable(0, name="globle_step", trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                          self.config['global']['max_grad_norm'])

        # optimizer = tf.train.GradientDescentOptimizer(lstm.lr)
        self.lr = tf.train.exponential_decay(self.config['global']['learning_rate'],self.global_step,1000,0.95,staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer.apply_gradients(zip(grads, tvars))
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        return train_op