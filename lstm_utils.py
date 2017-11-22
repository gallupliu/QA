# -*- coding: utf-8 -*-
# @Time    : 17-11-11 下午10:18
# @Author  : gallup
# @Email   : gallup-liu@hotmail.com
# @File    : lstm_utils.py
# @Software: PyCharm

import  tensorflow as tf


def rnn_cell(num_layers, num_hidden, cell_name, dropout):
    #
    # with tf.name_scope(cell_name+scope), tf.variable_scope(cell_name+scope):
    if cell_name == "gru":
        cells = [tf.contrib.rnn.GRUCell(num_hidden) for _ in range(num_layers)]
    elif cell_name == "lstm":
        cells = [tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True) for _ in
                 range(num_layers)]
    elif cell_name == "rnn":
        cell = [tf.contrib.rnn.RNNCell(num_hidden) for _ in range(num_layers)]
    # cells = tf.contrib.rnn.DropoutWrapper(cells)
    cells = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(cells), output_keep_prob=dropout)
    return cells

def transform_inputs(self, inputs, num_hidden, sequence_length):

    inputs = tf.transpose(inputs, [1, 0, 2])
    inputs = tf.reshape(inputs, [-1, num_hidden])
    inputs = tf.split(inputs, sequence_length, 0)
    return inputs