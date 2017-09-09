
import datetime
import os
import time
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
import tensorflow as tf
import data_helper
from bilstm import BILSTM

# 定义参数
trainingFile = "data/training.data"
developFile = "data/develop.data"
testingFile = "data/testing.data"
resultFile = "predictRst.score"
saveFile = "newModel/savedModel"
trainedModel = "trainedModel/savedModel"

#word2vec_model = KeyedVectors.load_word2vec_format('data/wiki.en.text.jian.vector', binary=True)
word2vec_model = Word2Vec.load('./data/wiki.en.text.jian.model')
data_set = data_helper.read_word_char('./data/training.data')
word_embed_dict = data_helper.generate_vocab(word2vec_model, data_set)
embeddings,word2idx = data_helper.generate_embeddings(50, word_embed_dict)


# Config函数
class CNN_Config(object):
    def __init__(self, vocab_size):
        # 输入序列(句子)长度
        self.sequence_length = 200
        # 循环数
        self.num_epochs = 100000
        # batch大小
        self.batch_size = 100
        # 词表大小
        self.vocab_size = vocab_size
        # 词向量大小
        self.embedding_size = 50
        # 不同类型的filter,相当于1-gram,2-gram,3-gram和5-gram
        self.filter_sizes = [1, 2, 3, 5]
        # 隐层大小
        self.hidden_size = 80
        # 每种filter的数量
        self.num_filters = 512
        # L2正则化,未用,没啥效果
        # 论文里给的是0.0001
        self.l2_reg_lambda = 0.
        # 弃权,未用,没啥效果
        self.keep_prob = 1.0
        # 学习率
        # 论文里给的是0.01
        self.lr = 0.01
        # margin
        # 论文里给的是0.009
        self.m = 0.05
        # 设定GPU的性质,允许将不能在GPU上处理的部分放到CPU
        # 设置log打印
        self.cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # 只占用20%的GPU内存
        self.cf.gpu_options.per_process_gpu_memory_fraction = 0.95

# Config函数
class BiLSTM_Config(object):
    def __init__(self, vocab_size):
        # 输入序列(句子)长度
        self.sequence_length = 100
        # 循环数
        self.num_epochs = 100
        # batch大小
        self.batch_size = 100
        # 词表大小
        self.vocab_size = vocab_size
        self.embeddings = embeddings
        # 词向量大小
        self.embedding_size = 50
        # 隐层大小
        self.rnn_size = 100
        # L2正则化,未用,没啥效果
        # 论文里给的是0.0001
        self.l2_reg_lambda = 0.
        # 弃权,未用,没啥效果
        self.dropout_keep_prob = 1.0
        # 学习率
        # 论文里给的是0.01
        self.lr = 0.01
        # margin
        # 论文里给的是0.009
        self.margin = 0.1

        self.max_grad_norm = 5

        #time step
        self.unrollSteps = 100  # 句子中的最大词汇数目
        # 设定GPU的性质,允许将不能在GPU上处理的部分放到CPU
        # 设置log打印
        self.cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # 只占用20%的GPU内存
        self.cf.gpu_options.per_process_gpu_memory_fraction = 0.95

    # 读取测试数据
print("正在载入测试数据，大约需要一分钟...")

config = BiLSTM_Config(len(embeddings))
qTest, aTest, _, qIdTest = data_helper.loadData(testingFile, word2idx, config.sequence_length)

def train_step(x1_batch,x2_batch,x3_batch, train_summary_op,train_summary_writer, model, sess):
    """
    A single training step
    """
    feed_dict = {
        model.input_questions: x1_batch,
        model.input_true_answers: x2_batch,
        model.input_false_answers: x3_batch,
        model.dropout_keep_prob: config.dropout_keep_prob
    }
    _, step, summaries, loss = sess.run(
        [model.train_op, model.global_step, train_summary_op,
         model.loss_op], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:.4f}".format(time_str, step,loss))
    train_summary_writer.add_summary(summaries, step)



def dev_step(x1_batch,x2_batch,dev_summary_op,dev_summary_writer, model, sess):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
        model.input_questions: x1_batch,
        model.input_true_answers: x2_batch,
        model.dropout_keep_prob: 1.0
    }
    step, summaries, loss= sess.run(
        [model.global_step, dev_summary_op,
         model.loss_op],feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:.4f}".format(time_str, step,loss))
    dev_summary_writer.add_summary(summaries, step)


def get_model(model_name,sess):
    model = None
    if model_name == 'match_lstm':
        # 配置文件

        model = BILSTM(config, sess)
    return model



def main(_):
    qTrain, aTrain, lTrain, qIdTrain = data_helper.loadData(trainingFile, word2idx, config.sequence_length, True)
    qTest, aTest, _, qIdTest = data_helper.loadData(testingFile, word2idx, config.sequence_length)
    qDevelop, aDevelop, lDevelop, qIdDevelop = data_helper.loadData(developFile, word2idx, config.sequence_length, True)
    trainQuestionCounts = qIdTrain[-1]
    for i in range(len(qIdDevelop)):
        qIdDevelop[i] += trainQuestionCounts



    # Training
    # 开始训练和测试
    with tf.device('/gpu:0'):
        with tf.Session(config=config.cf) as sess:
            # 建立网络
            model = get_model('match_lstm',sess)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join('..', 'runs', timestamp))
            print('Writing to {}\n'.format(out_dir))

            loss_summary = tf.summary.scalar(name='loss', tensor=model.loss_op)
            #accuracy_summary = tf.summary.scalar(name='accuracy', tensor=model.accuracy_op)
            #recall_summary = tf.summary.scalar(name='recall', tensor=model.recall_op)

            train_summary_op = tf.summary.merge(
                [loss_summary])
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            dev_summary_op = tf.summary.merge([loss_summary])
            dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            #max_to_keep设置保存模型的个数，默认为5


            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            tqs, tta, tfa = [], [], []
            for question, trueAnswer, falseAnswer in data_helper.trainingBatchIter(qTrain + qDevelop, aTrain + aDevelop,
                                                                                   lTrain + lDevelop,
                                                                                   qIdTrain + qIdDevelop,
                                                                                   config.batch_size):
                tqs.append(question), tta.append(trueAnswer), tfa.append(falseAnswer)
            for epoch in range(config.num_epochs):
                for question, trueAnswer, falseAnswer in zip(tqs, tta, tfa):
                    train_step(question, trueAnswer, falseAnswer, train_summary_op, train_summary_writer, model, sess)

                current_step = tf.train.global_step(sess, model.global_step)
                if current_step % config.num_epochs == 0:
                    print('\nEvaluation:')
                    dev_step( qTest, aTest, dev_summary_op,
                             dev_summary_writer, model, sess)
                    print('')
                if current_step % 1000 == 0:
                    path = saver.save(sess, save_path=checkpoint_prefix,
                                      global_step=model.global_step)
                    print('Saved model checkpoint to {}\n'.format(path))







if __name__ == '__main__':
    tf.app.run()