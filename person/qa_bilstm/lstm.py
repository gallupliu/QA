
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops

class LSTM(object):
    def __init__(self, batch_size, num_unroll_steps, embeddings, embedding_size, rnn_size, num_rnn_layers, max_grad_norm, l2_reg_lambda=0.0, adjust_weight=False,label_weight=[],is_training=True):
        # define input variable
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.adjust_weight = adjust_weight
        self.label_weight = label_weight
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.num_unroll_steps = num_unroll_steps
        self.max_grad_norm = max_grad_norm
        self.l2_reg_lambda = l2_reg_lambda
        self.is_training = is_training

        self.keep_prob = tf.placeholder(tf.float32, name="keep_drop")
        
        self.lr = tf.Variable(0.0,trainable=False)
        self.new_lr = tf.placeholder(tf.float32, shape=[],name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self.new_lr)

        self.ori_input_quests = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps])
        self.cand_input_quests = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps])
        self.neg_input_quests = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps])

        #build LSTM network
        fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, forget_bias=0.0, state_is_tuple=True)
        self.fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell] * self.num_rnn_layers, state_is_tuple=True)
        self.initial_fw_state = self.fw_cell.zero_state(self.batch_size, dtype=tf.float32)
        
        bw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, forget_bias=0.0, state_is_tuple=True)
        self.bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell] * self.num_rnn_layers, state_is_tuple=True)
        self.initial_bw_state = self.bw_cell.zero_state(self.batch_size, dtype=tf.float32)


        #embedding layer
        with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
            W = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
            self.ori_quests =tf.nn.embedding_lookup(W, self.ori_input_quests)
            self.cand_quests =tf.nn.embedding_lookup(W, self.cand_input_quests)
            self.neg_quests =tf.nn.embedding_lookup(W, self.neg_input_quests)

        #ori_quests = tf.nn.dropout(ori_quests, self.keep_prob)
        #cand_quests = tf.nn.dropout(cand_quests, self.keep_prob)
        #neg_quests = tf.nn.dropout(neg_quests, self.keep_prob)

	# forward  input(batch_size, num_unroll_steps, embedding_size)
        fw_ori_out, fw_cand_out, fw_neg_out = self.cal_rnn(self.fw_cell, self.initial_fw_state, self.num_unroll_steps, self.ori_quests, self.cand_quests, self.neg_quests, "fw_scope")

	# backward output  (num_unroll_steps, batch_size, embedding_size)
        self.ori_reverse_quests, self.cand_reverse_quests, self.neg_reverse_quests = tf.reverse(self.ori_quests, dims=[False, True, False]), tf.reverse(self.cand_quests, dims=[False, True, False]), tf.reverse(self.neg_quests, dims=[False, True, False])
        bw_ori_out, bw_cand_out, bw_neg_out = self.cal_rnn(self.bw_cell, self.initial_bw_state, self.num_unroll_steps, self.ori_reverse_quests, self.cand_reverse_quests, self.neg_reverse_quests, "bw_scope")

	# reverse bw_out input(num_unroll_step, batch_size, rnn_size) 
        out_bw_ori, out_bw_cand, out_bw_neg = tf.reverse(bw_ori_out, dims=[True, False, False]), tf.reverse(bw_cand_out, dims=[True, False, False]), tf.reverse(bw_neg_out, dims=[True, False, False])


	bw_ori = [tf.squeeze(each_bw_ori,[0]) for each_bw_ori in tf.split(0, self.num_unroll_steps, out_bw_ori)]
	bw_cand = [tf.squeeze(each_bw_cand,[0]) for each_bw_cand in tf.split(0, self.num_unroll_steps, out_bw_cand)]
	bw_neg = [tf.squeeze(each_bw_neg,[0]) for each_bw_neg in tf.split(0, self.num_unroll_steps, out_bw_neg)]

        #output(num_unroll_step, batch_size, 2*rnn_size)
        ori_out_put, cand_out_put, neg_out_put = self.concat_out(fw_ori_out, bw_ori), self.concat_out(fw_cand_out, bw_cand), self.concat_out(fw_neg_out, bw_neg)

	# ori_out_put(num_unroll_steps * batch_size * rnn_size) 
        with tf.name_scope("regulation_layer"):
            ori_out_put, cand_out_put, neg_out_put = tf.transpose(ori_out_put, perm=[1,2,0]), tf.transpose(cand_out_put, perm=[1,2,0]), tf.transpose(neg_out_put, perm=[1,2,0])
            ori_batch_output, cand_batch_output, neg_batch_output = [], [], []
            for sent_idx in range(self.batch_size):
	            ori_batch_output.append(tf.reduce_max(ori_out_put[sent_idx], 1))
	            cand_batch_output.append(tf.reduce_max(cand_out_put[sent_idx], 1))
	            neg_batch_output.append(tf.reduce_max(neg_out_put[sent_idx], 1))
        self.out_ori = tf.nn.tanh(ori_batch_output, name="tanh_ori")#(batch_size, 2 * rnn_size)
        self.out_cand = tf.nn.tanh(cand_batch_output, name="tanh_cand")
        self.out_neg = tf.nn.tanh(neg_batch_output, name="tanh_neg")

        # dropout
        #self.out_ori = tf.nn.dropout(self.out_ori, self.keep_prob)
        #self.out_cand = tf.nn.dropout(self.out_cand, self.keep_prob)
        #self.out_neg = tf.nn.dropout(self.out_neg, self.keep_prob)

        # cal cosine simulation
        self.ori_seq_len = tf.sqrt(tf.reduce_sum(tf.mul(self.out_ori, self.out_ori), 1), name="sqrt_ori")
        self.cand_seq_len = tf.sqrt(tf.reduce_sum(tf.mul(self.out_cand, self.out_cand), 1), name="sqrt_cand")
        self.neg_seq_len = tf.sqrt(tf.reduce_sum(tf.mul(self.out_neg, self.out_neg), 1), name="sqrt_neg")

        self.ori_cand_dist = tf.reduce_sum(tf.mul(self.out_ori, self.out_cand), 1, name="ori_cand")
        self.ori_neg_dist = tf.reduce_sum(tf.mul(self.out_ori, self.out_neg), 1, name="ori_neg")

        # cal the score
        with tf.name_scope("score"):
            self.ori_cand_score = tf.div(self.ori_cand_dist, tf.mul(self.ori_seq_len, self.cand_seq_len), name="score_positive")
            self.ori_neg_score = tf.div(self.ori_neg_dist, tf.mul(self.ori_seq_len, self.neg_seq_len), name="score_negative")

        # the target function 
        zero = tf.fill(tf.shape(self.ori_cand_score), 0.0)
        margin = tf.fill(tf.shape(self.ori_cand_score), 0.1)
        l2_loss = tf.constant(0.0)
        with tf.name_scope("loss"):
            losses = tf.maximum(zero, tf.sub(margin, tf.sub(self.ori_cand_score, self.ori_neg_score)))
            self.loss = tf.reduce_sum(losses) + self.l2_reg_lambda * l2_loss
        # cal accurancy
        with tf.name_scope("acc"):
            correct = tf.equal(zero, losses)
            self.acc = tf.reduce_mean(tf.cast(correct, "float"), name="acc")


    def cal_rnn(self, cell, _initial_state, num_unroll_steps, ori_quests, cand_quests, neg_quests, scope):
        ori_out_put=[]
        cand_out_put=[]
        neg_out_put=[]
        ori_state = _initial_state
        cand_state = _initial_state
        neg_state = _initial_state
        with tf.variable_scope(scope):
            for time_step in range(num_unroll_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (ori_cell_output, ori_state)=cell(ori_quests[:,time_step, :], ori_state)
                ori_out_put.append(ori_cell_output)
                
                tf.get_variable_scope().reuse_variables()
                (cand_cell_output, cand_state)=cell(cand_quests[:,time_step, :], cand_state)
                cand_out_put.append(cand_cell_output)

                tf.get_variable_scope().reuse_variables()
                (neg_cell_output, neg_state)=cell(neg_quests[:,time_step, :], neg_state)
                neg_out_put.append(neg_cell_output)
        return ori_out_put, cand_out_put, neg_out_put

    def concat_out(self, output_fw, output_bw):
        # Concat each of the forward/backward outputs
        flat_output_fw = nest.flatten(output_fw)
        flat_output_bw = nest.flatten(output_bw)
      
        flat_outputs = tuple(array_ops.concat(1, [fw, bw])
                             for fw, bw in zip(flat_output_fw, flat_output_bw))
      
        outputs = nest.pack_sequence_as(structure=output_fw,
                                   flat_sequence=flat_outputs)
        return outputs

    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self.new_lr:lr_value})
