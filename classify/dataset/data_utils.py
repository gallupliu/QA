# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: data_utils.py
@time: 2018/3/3 12:02


"""
import re
import os
import sys
import csv
import jieba
import logging
import multiprocessing
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.model_selection import train_test_split
import tensorflow as tf
from collections import defaultdict

EMBEDDING_SIZE = 50
def loadfile(outfile):
    neg=pd.read_csv('./data/neg.csv',header=None,index_col=None)
    pos=pd.read_csv('./data/pos.csv',header=None,index_col=None,error_bad_lines=False)
    neu=pd.read_csv('./data/neutral.csv', header=None, index_col=None)

    combined = np.concatenate((pos[0], neu[0], neg[0]))
    y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neu), dtype=int),
                        -1*np.ones(len(neg),dtype=int)))
    data_length = len(combined)
    print()
    text = []
    for i in range(data_length):
        text.append(clean_string(combined[i]))
    labels = []
    for i in range(len(pos)):
        labels.append([0,1,0])
    for i in range(len(neu)):
        labels.append([1,0,0])
    for i in range(len(neg)):
        labels.append([0,0,1])
    # assert  len(combined) == len(y)
    # f = open('./test_cut.txt','w',encoding="utf8")
    # with open(outfile,'w',newline='',encoding='utf-8') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['text', 'label'])
    #     for i in range(data_length):
    #         writer.writerow([combined[i],y[i]])
    #         f.write(clean_string(combined[i]))



    return text,labels


def clean_string(context):
    """
    字符串清洗和替换
    :param context:
    :return:
    """
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    context = re.sub('[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}', '<unk_email>', context)
    context = re.sub('(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', '<unk_ip>', context)
    context = re.sub('((?:http|ftp)s?://|(www|ftp)\.)[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)+([/?].*)?', '<unk_url>', context)
    context = re.sub('\d{4}-\d{1,2}-\d{1,2}','<unk_date>',context)
    context = re.sub('[\s+\.\!\/,$%^*:)(+\"\']+|[+!！，。？、~@#￥%……&*（）：-]',"",context)
    # words_list = ' '.join(jieba.cut(context))
    words_list = []
    for word in jieba.cut(context):
        words_list.append(word)
    return words_list


def train_word2vec():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # inp为输入语料, outp1 为输出模型, outp2为原始c版本word2vec的vector格式的模型
    fdir = './'
    inp = fdir + 'test_cut.txt'
    outp1 = fdir + 'wiki_50.model'
    outp2 = fdir + 'wiki_50.vector'

    # 训练skip-gram模型
    model = Word2Vec(LineSentence(inp), size=50, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    # 保存模型
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)

def load_embedding(infile,word2vec_path):
    words = set()
    fr = open(infile, "r", encoding="utf8")
    model = Word2Vec.load(word2vec_path)
    lines = fr.readlines()
    for line in lines:
        data = line.split(" ")
        for word in data:
            words.add(word)
    word_embed_dict = {}
    for word in words:
        if word in model:
            word_embed_dict[word] = model[word].tolist()

    index2word = ["<PAD>", '<UNK>'] + list(word_embed_dict.keys())
    word2index = dict([(y, x) for (x, y) in enumerate(index2word)])
    word_embeddings = [[0.0] * EMBEDDING_SIZE, [0.0] * EMBEDDING_SIZE]
    for _, word in enumerate(index2word[2:]):
        word_embeddings.append(word_embed_dict[word])
    return word2index,word_embeddings

def get_sentence_ids(text,word2idx,max_size=120):
    new_text = []
    for data in text:
        # new_data = []
        length = len(data)
        if len(data) < max_size:
            num =  max_size - length

            new_data = data + num * ["<PAD>"]

        else:
            new_data = data[:max_size]

        ids = [word2idx.get(word,1) for word in new_data]
        new_text.append(ids)
    return new_text


def count_length(data):
    dd = defaultdict(int)
    data_lengh = len(data)
    print(data_lengh,type(data))
    for i in range(data_lengh):
        length = len(data[i])
        if length in dd:

            dd[length] += 1
        else:
            dd[length] = 1

    new_dd = sorted(dd.items(), key=lambda d: d[0])

    count_len = 0
    max_length = 0
    for item in new_dd:
        print(item)
        count_len += item[1]
        if count_len / data_lengh > 0.90:
            max_length = item[0]
            break


    return max_length
if __name__ == '__main__':
    text,labels = loadfile('./data_with_label.csv')
    train_text,test_text,train_labels,test_labels = train_test_split(text,labels,test_size=0.1)
    # print(len(text),type(text))
    # max_length = count_length(text)
    # print(max_length)
    # train_word2vec()
    word2idx,vocab = load_embedding('./test_cut.txt','./wiki_50.model')
    # print(type(text))
    # print(list(word2idx.keys()))
    ids = get_sentence_ids(train_text, list(word2idx.keys()))
    dataset =  tf.data.Dataset.from_tensor_slices((ids, train_labels))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        print(sess.run(next_element))

    # print(len(word2idx))
    # print(word2idx["<PAD>"])
    # print(vocab[0])
    # print(word2idx["<UNK>"])
    # print(vocab[1])
