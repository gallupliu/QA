# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: evalution.py
@time: 2017/12/10 11:20


"""
import codecs
import numpy as np
import random
import math

class Evaluation(object):
    labels_dict = {}
    scores_dict = {}
    ACC_at1List = []
    APlist = []
    RRlist = []

    def __init__(self, data, scores):
        self.preprocess(data,scores)
        self.eval()


    def preprocess(self,data,scores):
        """
        :描述：将data中答案和scores按照字典形式对应
        :param data:list
        :param scores:
        :param metrics:
        """
        assert len(data) == len(scores)
        q_id = 0
        a_id = 0
        last_question = ''
        for id in range(len(data)):
            assert len(data[id]) == 3
            question = data[id][0]
            label = int(data[id][2])
            score = float(scores[id])
            if question != last_question:
                if id != 0:
                    q_id += 1
                a_id = 0
                last_question = question
            if not q_id in self.scores_dict:
                self.scores_dict[q_id] = {}
                self.labels_dict[q_id] = {}
            self.labels_dict[q_id][a_id] = label
            self.scores_dict[q_id][a_id] = score
            a_id += 1

    def eval(self,metrics):
        for qid, dinfo in self.scores_dict.items():
            y_pred = []
            y_true = []
            for did, s in dinfo.items():
                y_pred.append(s)
                y_true.append(self.labels_dict[qid][did])


    def eval_map(y_true, y_pred, rel_threshold=0):
        """
        :描述：计算平均准确率
        :param y_pred:
        :param rel_threshold:
        :return:
        """
        s = 0.
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        c = zip(y_true, y_pred)
        random.shuffle(c)
        c = sorted(c, key=lambda x: x[1], reverse=True)
        ipos = 0
        for j, (g, p) in enumerate(c):
            if g > rel_threshold:
                ipos += 1.
                s += ipos / (j + 1.)
        if ipos == 0:
            s = 0.
        else:
            s /= ipos
        return s

    def eval_ndcg(y_true, y_pred, k=10, rel_threshold=0.):
        """
        :描述：计算ndcg@n
        :param y_pred:
        :param k:
        :param rel_threshold:
        :return:
        """
        if k <= 0:
            return 0.
        s = 0.
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        c = zip(y_true, y_pred)
        random.shuffle(c)
        c_g = sorted(c, key=lambda x: x[0], reverse=True)
        c_p = sorted(c, key=lambda x: x[1], reverse=True)
        idcg = 0.
        ndcg = 0.
        for i, (g, p) in enumerate(c_g):
            if i >= k:
                break
            if g > rel_threshold:
                idcg += (math.pow(2., g) - 1.) / math.log(2. + i)
        for i, (g, p) in enumerate(c_p):
            if i >= k:
                break
            if g > rel_threshold:
                ndcg += (math.pow(2., g) - 1.) / math.log(2. + i)
        if idcg == 0.:
            return 0.
        else:
            return ndcg / idcg

    def eval_precision(y_true, y_pred, k=10, rel_threshold=0.):
        """
        :描述：计算precision@n
        :param y_pred:
        :param k:
        :param rel_threshold:
        :return:
        """
        if k <= 0:
            return 0.
        s = 0.
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        c = zip(y_true, y_pred)
        random.shuffle(c)
        c = sorted(c, key=lambda x: x[1], reverse=True)
        ipos = 0
        precision = 0.
        for i, (g, p) in enumerate(c):
            if i >= k:
                break
            if g > rel_threshold:
                precision += 1
        precision /= k
        return precision



if __name__ == '__main__':
    QApairFile='../../data/develop.data'
    scoreFile='predictRst.score'
    outputFile = ''
    qaPairLines = codecs.open(QApairFile, 'r', 'utf-8').readlines()
    data = []
    score_dict = {}
    data_dict = {}
    qIndex = 0
    questiones =set()
    lastQuestion = ""
    for idx in range(len(qaPairLines)):
        qaLine = qaPairLines[idx].strip()
        qaLineArr = qaLine.split('\t')
        assert len(qaLineArr) == 3
        question = qaLineArr[0]
        questiones.add(question)
        #             answer=qaLineArr[1]
        label = int(qaLineArr[2])
        if question != lastQuestion:
            if idx != 0:
                qIndex += 1
            aIndex = 0
            lastQuestion = question
        if not qIndex in score_dict:
            score_dict[qIndex] = {}
            data_dict[qIndex] = {}
        data_dict[qIndex][aIndex] = label
        aIndex += 1
    # for item in data_dict:
    #     print(item)
    print(data_dict)
    print(len(data_dict))
    print(len(questiones))