# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: main.py
@time: 2017/12/8 22:06


"""
import os
import sys
import logging
import datetime
import time
import json
import tensorflow as tf
import operator
import argparse

from preprocess.data_helper import load_train_data, load_test_data, load_embedding, batch_iter
from models.bilstm import BiLSTM
from metrics.evalution import Evaluation

# ------------------------- define parameter -----------------------------
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_options", 0.9, "use memory rate")

FLAGS = tf.flags.FLAGS
# ----------------------------- define parameter end ----------------------------------

# ----------------------------- define a logger -------------------------------
logger = logging.getLogger("execute")
logger.setLevel(logging.INFO)

fh = logging.FileHandler("./run.log", mode="w")
fh.setLevel(logging.INFO)

fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)

fh.setFormatter(formatter)
logger.addHandler(fh)
# ----------------------------- define a logger end ----------------------------------



# ----------------------------------- execute train model ---------------------------------
def run_step(sess, ori_batch, cand_batch, neg_batch, model, dropout=1.):
    start_time = time.time()
    feed_dict = {
        model.query: ori_batch,
        model.input_left: cand_batch,
        model.input_right: neg_batch,
        model.keep_prob: dropout
    }

    _, step, ori_cand_score, ori_neg_score, cur_loss, cur_acc = sess.run(
        [model.train_op, model.global_step, model.ori_cand, model.ori_neg, model.loss, model.acc], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    right, wrong, score = [0.0] * 3
    for i in range(0, len(ori_batch)):
        if ori_cand_score[i] > 0.55 and ori_neg_score[i] < 0.4:
            right += 1.0
        else:
            wrong += 1.0
        score += ori_cand_score[i] - ori_neg_score[i]
    time_elapsed = time.time() - start_time
    logger.info("%s: step %s, loss %s, acc %s, score %s, wrong %s, %6.7f secs/batch" % (
    time_str, step, cur_loss, cur_acc, score, wrong, time_elapsed))

    return cur_loss, ori_cand_score


def valid_run_step(sess, ori_batch, cand_batch, model, dropout=1.):
    feed_dict = {
        model.query: ori_batch,
        model.input_left: cand_batch,
        model.keep_prob: dropout
    }

    step, ori_cand_score = sess.run([model.global_step, model.ori_cand], feed_dict)

    return ori_cand_score


# ---------------------------------- execute train model end --------------------------------------

def cal_acc(labels, results, total_ori_cand):
    if len(labels) == len(results) == len(total_ori_cand):
        retdict = {}
        for label, result, ori_cand in zip(labels, results, total_ori_cand):
            if result not in retdict:
                retdict[result] = []
            retdict[result].append((ori_cand, label))

        correct = 0
        for key, value in retdict.items():
            value.sort(key=operator.itemgetter(0), reverse=True)
            score, flag = value[0]
            if flag == 1:
                correct += 1
        return 1. * correct / len(retdict)
    else:
        logger.info("data error")
        return 0


# ---------------------------------- execute valid model ------------------------------------------
def valid_model(sess, model, valid_ori_quests, valid_cand_quests, labels, results,config):
    logger.info("start to validate model")
    total_ori_cand = []
    for ori_valid, cand_valid, neg_valid in batch_iter(valid_ori_quests, valid_cand_quests, config['inputs']['train']['batch_size'], 1,
                                                       is_valid=True):
        ori_cand = valid_run_step(sess, ori_valid, cand_valid, model)
        total_ori_cand.extend(ori_cand)

    data_len = len(total_ori_cand)
    data = []
    for i in range(data_len):
        data.append([valid_ori_quests[i], valid_cand_quests[i], labels[i]])

    evalution = Evaluation(data, results)
    acc = cal_acc(labels[:data_len], results[:data_len], total_ori_cand)


    timestr = datetime.datetime.now().isoformat()
    logger.info("%s, evaluation mrr:%s,map:%s,test_map:%s,test_mrr:%s,acc:%s:" % (timestr, evalution.MRR(),evalution.MAP(),evalution.map_1,evalution.mrr_1,acc))


# ---------------------------------- execute valid model end --------------------------------------


def get_model(config,embedding):
    model = None
    print(config['model_name'])
    try:
        if config['model_name'] == "BiLSTM":
            model = BiLSTM(config, embedding)
    except Exception as e:
        logging.error("load model Exception", e)
        exit()

    return model

def train(config):
    logging.info("start load data")
    embedding, word2idx, idx2word = load_embedding(config['inputs']['share']['embed_file'], config['inputs']['share']['embed_size'])
    ori_quests, cand_quests = load_train_data(config['inputs']['train']['relation_file'], word2idx,
                                              config['inputs']['share']['text1_maxlen'])

    valid_ori_quests, valid_cand_quests, valid_labels, valid_results = load_test_data(config['inputs']['valid']['relation_file'],
                                                                                      word2idx,
                                                                                      config['inputs']['share']['text1_maxlen'])


    logging.info("start train")
    with tf.Graph().as_default():
        with tf.device("/cpu:0"):
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_options)
            session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)
            with tf.Session(config=session_conf).as_default() as sess:
                model = get_model(config, embedding)

                sess.run(tf.global_variables_initializer())

                for epoch in range(config['inputs']['train']['epoches']):
                    # cur_lr = FLAGS.lr / (epoch + 1)
                    # model.assign_new_lr(sess, cur_lr)
                    # logger.info("current learning ratio:" + str(cur_lr))
                    for ori_train, cand_train, neg_train in batch_iter(ori_quests, cand_quests, config['inputs']['train']['batch_size'],
                                                                       epoches=1):
                        run_step(sess, ori_train, cand_train, neg_train, model)
                        cur_step = tf.train.global_step(sess, model.global_step)

                        if cur_step % 100 == 0 and cur_step != 0:
                            valid_model(sess, model, valid_ori_quests, valid_cand_quests, valid_labels, valid_results,config)
                # valid_model(sess, model, test_ori_quests, test_cand_quests, labels, results)
    # ---------------------------------- end train -----------------------------------

def predict(config):
    if os.path.exists(config['model']['model_path']):
        try:
            saver = tf.train.import_meta_graph("Model/model.ckpt.meta")
            with tf.Session() as sess:
                saver.restore(sess, config['model']['model_path'])
        except Exception as e:
            logging.info("model not found",e)



def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', help='Phase: Can be train or predict, the default value is train.')
    parser.add_argument('--model_file', default='./configs/insurance/bilstm_config', help='Model_file: modelZoo model file for the chosen model.')
    args = parser.parse_args()
    model_file =  args.model_file
    with open(model_file, 'r') as f:
        config = json.load(f)

    # test_ori_quests, test_cand_quests, labels, results = load_test_data(config['inputs']['test']['relation_file'], word2idx,
    #                                                                     config['inputs']['share']['text1_maxlen'])

    if args.phase == 'train':
        train(config)
    elif args.phase == 'predict':
        predict(config)
    elif args.phase == "export":
        pass
    else:
        print('Phase Error.', end='\n')
    return

if __name__=='__main__':
    main(sys.argv)
