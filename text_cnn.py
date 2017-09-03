# -*- coding:utf-8 -*-

import tensorflow as tf


def linear(input_, output_size, scope=None):
    """
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    :param input_: a tensor or a list of 2D, batch x n, Tensors.
    :param output_size: int, second dimension of W[i].
    :param scope: VariableScope for the created subgraph; defaults to "Linear".
    :returns: A 2D Tensor with shape [batch x output_size] equal to \
                sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    :raises: ValueError, if some of the arguments has unspecified or wrong shape.
    """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """
    Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))
            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)
            output = t * g + (1. - t) * input_
            input_ = output

    return output


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, pretrained_embedding=None):

        # Placeholders for input, output and dropout
        self.input_x_front = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_front")
        self.input_x_behind = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_behind")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Vector = load_word2vec(TEXT_DATA_DIR, WORD2VEC_DIR, DICTIONARY_DIR, vocab_size, embedding_size)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # 原采用的是随机生成正态分布的词向量。
            # Vector 是通过自己的语料库训练而得到的词向量。
            # input_x_front 和 input_x_behind 共用词向量。
            if pretrained_embedding is None:
                self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            else:
                self.W = tf.Variable(pretrained_embedding, name="W", trainable=True)
                self.W = tf.cast(self.W, tf.float32)
            self.embedded_chars_front = tf.nn.embedding_lookup(self.W, self.input_x_front)
            self.embedded_chars_behind = tf.nn.embedding_lookup(self.W, self.input_x_behind)
            self.embedded_chars_expanded_front = tf.expand_dims(self.embedded_chars_front, -1)
            self.embedded_chars_expanded_behind = tf.expand_dims(self.embedded_chars_behind, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_front = []
        pooled_outputs_behind = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv_front = tf.nn.conv2d(
                    self.embedded_chars_expanded_front,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_front")

                conv_behind = tf.nn.conv2d(
                    self.embedded_chars_expanded_behind,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_behind")

                # Apply nonlinearity
                h_front = tf.nn.relu(tf.nn.bias_add(conv_front, b), name="relu_front")
                h_behind = tf.nn.relu(tf.nn.bias_add(conv_behind, b), name="relu_behind")
                # Maxpooling over the outputs
                pooled_front = tf.nn.max_pool(
                    h_front,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool_front")

                pooled_behind = tf.nn.max_pool(
                    h_behind,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool_behind")

                pooled_outputs_front.append(pooled_front)
                pooled_outputs_behind.append(pooled_behind)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_front = tf.concat(pooled_outputs_front, 3)
        self.h_pool_behind = tf.concat(pooled_outputs_behind, 3)
        self.h_pool_flat_front = tf.reshape(self.h_pool_front, [-1, num_filters_total])
        self.h_pool_flat_behind = tf.reshape(self.h_pool_behind, [-1, num_filters_total])

        self.h_pool_flat_combine = tf.concat([self.h_pool_flat_front, self.h_pool_flat_behind], 1)

        # Add highway
        with tf.name_scope("highway"):
            self.h_highway = highway(self.h_pool_flat_combine, self.h_pool_flat_combine.get_shape()[1], 1, 0)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total * 2, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.softmaxScores = tf.nn.softmax(self.scores, name="softmaxScores")
            self.sigmoidScores = tf.nn.sigmoid(self.scores, name="sigmoidScores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.topKPreds = tf.nn.top_k(self.softmaxScores, k=1, sorted=True, name="topKPreds")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # AUC
        with tf.name_scope("AUC"):
            self.AUC = tf.contrib.metrics.streaming_auc(self.softmaxScores, self.input_y)