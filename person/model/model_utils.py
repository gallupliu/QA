# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: model_utils.py
@time: 2017/11/29 21:49


"""


import  tensorflow as tf

def rnn_cell(cell_name,num_layers, num_hidden,  dropout):
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

def single_rnn(inputs,cell_name,num_hidden,num_layers,dropout,scope):
    cells = rnn_cell(cell_name,num_hidden,num_layers,dropout)

    with tf.name_scope("RNN_"+cell_name+scope),tf.variable_scope("RNN_"+cell_name+scope):
        outputs, output_states_= tf.nn.dynamic_rnn(cells,inputs,time_major=False,dtype=tf.float32)
    return outputs


def bi_rnn(inputs,cell_name,num_hidden,num_layers,dropout,scope):


    with tf.name_scope("fw" + cell_name + scope), tf.variable_scope("fw" + cell_name + scope):
        lstm_fw_cell_m = rnn_cell(cell_name,num_layers, num_hidden,  dropout)

    with tf.name_scope("bw" + cell_name + scope), tf.variable_scope("bw" + cell_name + scope):
        lstm_bw_cell_m = rnn_cell(cell_name,num_layers, num_hidden,  dropout)

    with tf.name_scope("bi" + cell_name + scope), tf.variable_scope("b" + cell_name + scope):
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m,lstm_bw_cell_m,inputs,dtype=tf.float32)

    return outputs

