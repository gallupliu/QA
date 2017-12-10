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

class Evaluator(object):
    data_dict = {}
    score_dict = {}
    ACC_at1List = []
    APlist = []
    RRlist = []

    def __init__(self, data, scores,metrics):
        self.preprocess(data,scores,metrics)

    def _to_list(x):
        if isinstance(x, list):
            return x
        return [x]

    def zipped(self, y_true, y_pred):
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        c = zip(y_true, y_pred)
        random.shuffle(c)
        return c


    def preprocess(self,data,scores,metrics):
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
            answer = data[id][1]
            label = int(data[id][2])
            score = float(scores[id])
            if question != last_question:
                if id != 0:
                    q_id += 1
                a_id = 0
                last_question = question
            if not q_id in self.score_dict:
                self.score_dict[q_id] = {}
                self.data_dict[q_id] = {}
            self.data_dict[q_id][a_id] = label
            self.score_dict[q_id][a_id] = score
            a_id += 1
    def calculate_test(self,k):
        right_num = 0 #precision
        rank_index = 0
        cur_list = [] #mrr map
        for q_id, score_list in self.score_dict.items():
            label_list = self.data_dict[q_id]
            assert len(label_list) == len(score_list)
            ranked_list = sorted(zip(label_list, score_list),key=lambda x:x[1], reverse=True)
            length = len(ranked_list)
            for i in range(length):
                label = label_list[i]
                rank_index += 1

                if label == 1:
                    right_num += 1
                    p = float(right_num) / rank_index
                    cur_list.append(p)

                if i >= k:
                    break
            if len(cur_list) > 0 and len(cur_list) != len(ranked_list):
                self.RRlist.append(cur_list[0])
                self.APlist.append(float(sum(cur_list)) / len(cur_list))
            else:
                self.ACC_at1List.pop()


    def MRR(self):
        return float(sum(self.RRlist)) / len(self.RRlist)

    def MAP(self):
        return float(sum(self.APlist)) / len(self.APlist)

    def test(self):
        """
        https://github.com/SunflowerPKU/WikiQA-CNN/blob/master/src/eval.py
        :return:
        """
        MAP = 0.0
        MRR = 0.0
        for q_id, score_list in self.score_dict.items():
            label_list = self.data_dict[q_id]
            assert len(label_list) == len(score_list)
            ranked_list = sorted(zip(label_list, score_list),key=lambda x:x[1], reverse=True)
            correct = 0
            total = 0
            AP = 0.0
            mrr_mark = False
            for i in range(len(ranked_list)):
                label = label_list[i]
                # compute MRR
                if (label == '1' or label == 1) and mrr_mark == False:
                    MRR += 1.0 / float(i + 1)
                    mrr_mark = True
                # compute MAP
                total += 1
                if label == '1' or label == 1:
                    correct += 1
                    AP += float(correct) / float(total)
            AP /= float(correct)
            MAP += AP

        MAP /= float(len(score_dict))
        MRR /= float(len(score_dict))
        return MAP, MRR

    def Precision_at_k(self,k):
        """
        https://github.com/SunflowerPKU/WikiQA-CNN/blob/master/src/eval.py
        :return:
        """
        MAP = 0.0
        MRR = 0.0
        for q_id, score_list in self.score_dict.items():
            label_list = self.data_dict[q_id]
            assert len(label_list) == len(score_list)
            ranked_list = sorted(zip(label_list, score_list),key=lambda x:x[1], reverse=True)
            correct = 0
            total = 0
            for i in range(len(ranked_list)):
                label = label_list[i]
                if sum(label) > 1:
                    total += 1

                if (label == '1' or label == 1):
                    correct += 1
                if i >= k:
                    break
        return MAP, MRR

    def eval_ndcg(y_true, y_pred, k=10, rel_threshold=0.):
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


    def evaluate(QApairFile, scoreFile, outputFile='evaluation.score'):
        testor = Evaluator(QApairFile, scoreFile)
        testor.calculate()
        print("MRR:%f \t MAP:%f \t ACC@1:%f\n" % (testor.MRR(), testor.MAP(), testor.ACC_at_1()))
        if outputFile != '':
            fw = open(outputFile, 'a')
            fw.write('%f \t %f \t %f\n' % (testor.MRR(), testor.MAP(), testor.ACC_at_1()))



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