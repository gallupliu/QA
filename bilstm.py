import tensorflow as tf
import numpy as np


class BILSTM(object):
    """
    中文问答系统使用的LSTM网络结构
    """

    def __init__(self, config, sess):

        self.batchSize = config.batch_size
        self.embeddings = config.embeddings
        print(len(self.embeddings))
        self.embedding_size = config.embedding_size
        self.rnnSize = config.rnn_size
        self.margin = config.margin

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="keep_drop")
        self.input_questions = tf.placeholder(tf.int32, shape=[None, config.sequence_length])
        self.input_true_answers = tf.placeholder(tf.int32, shape=[None, config.sequence_length])
        self.input_false_answers = tf.placeholder(tf.int32, shape=[None, config.sequence_length])
        self.inputTestQuestions = tf.placeholder(tf.int32, shape=[None, config.sequence_length])
        self.inputTestAnswers = tf.placeholder(tf.int32, shape=[None, config.sequence_length])

        # 设置word embedding层
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            embeddings = tf.get_variable(name='word_embedding',
                                                   shape=[config.vocab_size, config.embedding_size],
                                                   initializer=tf.constant_initializer(self.embeddings),
                                                   trainable=False)

            questions = tf.nn.embedding_lookup(embeddings, self.input_questions)
            trueAnswers = tf.nn.embedding_lookup(embeddings, self.input_true_answers)
            falseAnswers = tf.nn.embedding_lookup(embeddings, self.input_false_answers)

            #testQuestions = tf.nn.embedding_lookup(embeddings, self.inputTestQuestions)
            #testAnswers = tf.nn.embedding_lookup(embeddings, self.inputTestAnswers)

        # 建立LSTM网络
        with tf.variable_scope("LSTM_scope", reuse=None):
            question1 = self.biLSTMCell(questions, self.rnnSize)
            question2 = tf.nn.tanh(self.max_pooling(question1))
        with tf.variable_scope("LSTM_scope", reuse=True):
            trueAnswer1 = self.biLSTMCell(trueAnswers, self.rnnSize)
            trueAnswer2 = tf.nn.tanh(self.max_pooling(trueAnswer1))
            falseAnswer1 = self.biLSTMCell(falseAnswers, self.rnnSize)
            falseAnswer2 = tf.nn.tanh(self.max_pooling(falseAnswer1))

            #testQuestion1 = self.biLSTMCell(testQuestions, self.rnnSize)
            #testQuestion2 = tf.nn.tanh(self.max_pooling(testQuestion1))
            #testAnswer1 = self.biLSTMCell(testAnswers, self.rnnSize)
            #testAnswer2 = tf.nn.tanh(self.max_pooling(testAnswer1))

        self.trueCosSim = self.getCosineSimilarity(question2, trueAnswer2)
        self.falseCosSim = self.getCosineSimilarity(question2, falseAnswer2)
        self.loss_op = self.getLoss(self.trueCosSim, self.falseCosSim, self.margin)
        self.train_op = self.train_op(self.loss_op)

        #self.result = self.getCosineSimilarity(testQuestion2, testAnswer2)


    def biLSTMCell(self,x, hiddenSize):
        input_x = tf.transpose(x, [1, 0, 2])
        input_x = tf.unstack(input_x)
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
        output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_x, dtype=tf.float32)
        output = tf.stack(output)
        output = tf.transpose(output, [1, 0, 2])
        return output

    def getCosineSimilarity(self,q, a):
        q1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        a1 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
        mul = tf.reduce_sum(tf.multiply(q, a), 1)
        cosSim = tf.div(mul, tf.multiply(q1, a1))
        return cosSim

    def max_pooling(self,lstm_out):
        height = int(lstm_out.get_shape()[1])
        width = int(lstm_out.get_shape()[2])
        lstm_out = tf.expand_dims(lstm_out, -1)
        output = tf.nn.max_pool(lstm_out, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output, [-1, width])
        return output

    def getLoss(self,trueCosSim, falseCosSim, margin):
        zero = tf.fill(tf.shape(trueCosSim), 0.0)
        tfMargin = tf.fill(tf.shape(trueCosSim), margin)
        with tf.name_scope("loss"):
            losses = tf.maximum(zero, tf.subtract(tfMargin, tf.subtract(trueCosSim, falseCosSim)))
            loss = tf.reduce_sum(losses)
        return loss


    def train_op(self,loss):
        with tf.variable_scope("trian"):
            self.global_step = tf.get_variable(shape=[],
                                               initializer=tf.constant_initializer(0),
                                               dtype=tf.int32,
                                               name='global_step')
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        #optimizer.apply_gradients(zip(grads, tvars))
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        #optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr, beta1=0.9, beta2=0.999)
        #train_op = optimizer.minimize(loss, global_step=self.global_step)
        return train_op

