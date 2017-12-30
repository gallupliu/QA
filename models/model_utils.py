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

def text_cnn(inputs,filter_sizes,num_filters,embedding_size,sequence_length,dropout_keep_prob=1.0):
    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]

            # Create variable named "weights".
            weights = tf.get_variable("weights", filter_shape,
                                      initializer=tf.random_normal_initializer())
            # Create variable named "biases".
            biases = tf.get_variable("biases", [num_filters],
                                     initializer=tf.constant_intializer(0.0))
            conv = tf.nn.conv2d(
                inputs,
                weights,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv,  biases), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Add dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    return h_drop




def rnn_cell(cell_name,num_layers, num_hidden,  dropout):
    #
    # with tf.name_scope(cell_name+scope), tf.variable_scope(cell_name+scope):
    if cell_name == "gru":
        cells = [tf.contrib.rnn.GRUCell(num_hidden) for _ in range(num_layers)]
    elif cell_name == "lstm":
        cells = [tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True) for _ in
                 range(num_layers)]
    elif cell_name == "rnn":
        cells = [tf.contrib.rnn.RNNCell(num_hidden) for _ in range(num_layers)]
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

