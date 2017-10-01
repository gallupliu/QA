# coding:utf-8

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell 
from utils import ortho_weight, uniform_weight
## define lstm model and reture related features


# return n outputs of the n lstm cells
def biLSTM(x, hidden_size):
	# biLSTM��
	# ���ܣ����bidirectional_lstm����
	# ������
	# 	x: [batch, height, width]   / [batch, step, embedding_size]
	# 	hidden_size: lstm���ز�ڵ����
	# �����
	# 	output: [batch, height, 2*hidden_size]  / [batch, step, 2*hidden_size]

	# input transformation
	input_x = tf.transpose(x, [1, 0, 2])
	# input_x = tf.reshape(input_x, [-1, w])
	# input_x = tf.split(0, h, input_x)
	input_x = tf.unpack(input_x)

	# define the forward and backward lstm cells
	lstm_fw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
	lstm_bw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
	output, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_x, dtype=tf.float32)

	# output transformation to the original tensor type
	output = tf.pack(output)
	output = tf.transpose(output, [1, 0, 2])
	return output

def LSTM(input_x, rnn_size, batch_size):#input(batch_size, steps, embedding_size)
    num_steps = int(input_x.get_shape()[1])
    embedding_size = int(input_x.get_shape()[2])
    #define parameter
    W = tf.get_variable("W", initializer=tf.concat(1, [uniform_weight(embedding_size, rnn_size), uniform_weight(embedding_size, rnn_size)]))
    U = tf.get_variable("U", initializer=tf.concat(1, [ortho_weight(rnn_size), ortho_weight(rnn_size)]))
    b = tf.get_variable("b", initializer=tf.zeros([2 * rnn_size]))
    Wx = tf.get_variable("Wx", initializer=uniform_weight(embedding_size, rnn_size))
    Ux = tf.get_variable("Ux", initializer=ortho_weight(rnn_size))
    bx = tf.get_variable("bx", initializer=tf.zeros([rnn_size]))
    h_ = tf.zeros([batch_size, rnn_size])
    one = tf.fill([batch_size, rnn_size], 1.)
    state_below = tf.transpose(tf.batch_matmul(input_x, tf.tile(tf.reshape(W, [1, embedding_size, 2 * rnn_size]), [batch_size, 1, 1])) + b, perm=[1, 0, 2])
    state_belowx = tf.transpose(tf.batch_matmul(input_x, tf.tile(tf.reshape(Wx, [1, embedding_size, rnn_size]), [batch_size, 1, 1])) + bx, perm=[1, 0, 2])#(steps, batch_size, rnn_size)
    output = []#(steps, batch_size, rnn_size)
    with tf.variable_scope("GRU"):
        for time_step in range(num_steps):
            preact = tf.matmul(h_, U)
            preact = tf.add(preact, state_below[time_step])
            
            r = tf.nn.sigmoid(_slice(preact, 0, rnn_size))
            u = tf.nn.sigmoid(_slice(preact, 1, rnn_size))

            preactx = tf.matmul(h_, Ux)
            preactx = tf.mul(preactx, r)
            preactx = tf.add(preactx, state_belowx[time_step])
            h = tf.tanh(preactx)

            h_ = tf.add(tf.mul(u, h_), tf.mul(tf.sub(one, u), h))
            output.append(h_)
    output = tf.transpose(output, perm=[1, 0, 2])
    return output#(batch_size, steps, rnn_size)

def _slice(_x, n, dim):
    if len(_x.get_shape()) == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    return _x[:, n*dim:(n+1)*dim]
