# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: data_utils.py
@time: 2018/2/8 20:25


"""
import re
import csv
import jieba
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.file_utils import serialize,deserialize


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
    #words_list = [''.join(word) for word in jieba.cut(context)]
    words_list = []
    for word in jieba.cut(context):
        words_list.append(word)
    return words_list



def get_words_dict():
    reader = open('./test','r',encoding='utf-8')
    lines = reader.readlines()
    words_dict = set()
    query = []
    answer = []
    for line in lines:
        data = line.split('\\t')
        print(clean_string(data[1]))
        query.append(clean_string(data[0]))
        answer.append(clean_string(data[1]))
        for word in clean_string(line):
            words_dict.add(word)

    words = []
    for word in words_dict:

        words.append(word)
    print(type(query),type(answer))
    return np.asarray(words),np.asarray(query),np.asarray(answer)



def get_word2index(mapping,sentence):
    """
    对分词后的list进行word2vec映射
    :param mapping:
    :param sentence:
    :return:
    """
    mapping_strings = tf.constant(mapping)
    # sentence_tensor = tf.constant(sentence)
    ids = tf.contrib.lookup.string_to_index(sentence,mapping_strings,default_value=-1)

    # with tf.Session() as sess:
    #     tf.tables_initializer().run()
    #     print(sess.run(ids))
    return ids

def gen_data():
    import itertools

    def gen():
        for i in itertools.count(1):
            yield (i, [1] * i)

    ds = tf.data.Dataset.from_generator(
        gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))
    value = ds.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        print(sess.run(value))  # (1, array([1]))
        print(sess.run(value))  # (2, array([1, 1]))

def read_csv_file(filename):
    """
    从csv生成dataset
    :param filename:
    :return:
    """
    raise NotImplementedError()


def read_text_line(filename):
    """
    直接读取原始数据生成dataset
    :param filename:
    :return:
    """
    raise NotImplementedError()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def write_tfrecord(infile,outfile):
    data = pd.read_csv(infile,encoding='utf-8')
    length = len(data)
    print(data)

    with tf.python_io.TFRecordWriter(outfile) as writer:
        for i in range(length):
            qid = data['qid'][i]

            query = clean_string(data['query'][i])
            print(query)
            answer = clean_string(data['answer'][i])
            label = data['label'][i]

            example = tf.train.Example(features=tf.train.Features(feature={
                "qid": tf.train.Feature(int64_list=tf.train.Int64List(value=[qid])),
                'query': _bytes_feature(query),
                'answer': _bytes_feature(answer),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            writer.write(example.SerializeToString())


def read_tfrecord(infile):
    """
    直接读取tfreocord生成dataset
    :return:
    """
    # for serialized_example in tf.python_io.tf_record_iterator(infile):
    #     # Get serialized example from file
    #     example = tf.train.Example()
    #     example.ParseFromString(serialized_example)
    #     qid = example.features.feature["qid"].int64_list.value
    #     query = example.features.feature["query"].bytes_list.value
    #     label = example.features.feature["answer"].bytes_list.value
    #     answer = example.features.feature["label"].int64_list.value
    #     print("qid: {}, query:{},answer:{},label: {}".format(qid,query,answer,label))

    def _parse_fn(proto):
        features = {
            "qid": tf.FixedLenFeature([1], tf.int64),
            "query": tf.VarLenFeature(tf.string),
            "answer": tf.VarLenFeature(tf.string),
            "label": tf.FixedLenFeature([1], tf.int64)}

        parsed_features = tf.parse_example(proto, features)

        query =  tf.sparse_tensor_to_dense(parsed_features["query"])
        answer =  tf.sparse_tensor_to_dense(parsed_features["answer"])
        label =  tf.sparse_tensor_to_dense(parsed_features["label"])
        print(query,answer,label[0])
        return parsed_features



    dataset = tf.data.TFRecordDataset(infile)

    dataset = dataset.map(_parse_fn)
    dataset = dataset.batch(2)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    sess = tf.Session()
    print(sess.run( features))


def convert_txt_csv(outfile):
    # data = pd.read_csv(infile,'utf-8',header=None)

    with open(outfile,'w',newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['qid', 'query', 'answer', 'label'])
        reader = open('./test', 'r', encoding='utf-8')
        lines = reader.readlines()
        last = ''
        qid = 0
        for line in lines:
            data = line.replace("\n","").replace(" ","").split('\\t')
            print(len(data))
            if last != data[0]:
                qid += 1
                last = data[0]
            if data[2]=='0':
                writer.writerow([qid,data[0],data[1],0])
            else:
                writer.writerow([qid, data[0], data[1], 1])
            print(data)


def train_input_fn(infile):
    data = pd.read_csv(infile,encoding='utf-8')
    """生成训练数据"""
    def gen():
        pos = {}
        neg = {}

        for item in data:
            if item[3] == 1:
                if item[1] in pos:
                    pos[item[0]].append(item)
                else:
                    pos[item[0]] = []
            else:
                if item[1] in neg:
                    neg[item[0]].append(item)
                else:
                    neg[item[0]] = []
            #分词，获取index编码


def write_npy(infile,outfile):
    data = pd.read_csv(infile,encoding='utf-8')
    data_length = len(data)
    new_data = []

    for i in range(data_length):
        print(clean_string(data['query'][i]))
        print(data['qid'][i], clean_string(data['query'][i]), clean_string(data['answer'][i]), data['label'][i])
        new_data.append([data['qid'][i], np.asarray(clean_string(data['query'][i])), np.asarray(clean_string(data['answer'][i])), data['label'][i]])
    serialize(new_data,outfile)


if __name__ == "__main__":
    #1、从预生成的ids中
    # write_npy('./test.csv','./test.bin')
    words, query, answer = get_words_dict()
    data = deserialize('./test.bin')
    data_length = len(data)
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(words), num_oov_buckets=1, default_value=-1)
    with tf.Session() as  sess:
        tf.tables_initializer().run()
        ids = table.lookup(tf.constant(data[:,i]))
        for i in range(data_length):



            print(data[i][1])



    # write_tfrecord('./test.csv', './test.tfrecord')
    # read_tfrecord('./test.tfrecord')
    # convert_txt_csv('./test.csv')

    # words, query, answer = get_words_dict()
    #
    # filenames = ['./list']
    # dataset = tf.data.TextLineDataset(filenames=filenames)
    # #一次只生成一条数据
    # iterator = dataset.make_one_shot_iterator()
    # # iterator_1 = dataset.make_initializable_iterator()
    #
    # batch = iterator.get_next()
    # # batch_1 = iterator_1.get_next()
    # ids = tf.contrib.lookup.string_to_index(tf.constant( [['芝加哥','国际','电影节','最佳','女主角', '是', '谁'], ['2002', '年', '来自', '30', '多个', '国家', '和', '地区', '的', '90', '多部', '故事片', '40', '多部', '短片', '参加', '了', '电影节', '吸引', '了', '世界各地', '的', '6', '万多', '观众']]),tf.constant(words), default_value=-1)
    # #ids = tf.contrib.lookup.string_to_index(tf.constant(['芝加哥', '国际', '电影节', '最佳', '女主角', '是', '谁']), tf.constant(words), default_value=-1)
    # with tf.Session() as sess:
    #     tf.tables_initializer().run()
    #     # sess.run(iterator_1.initializer)
    #     for i  in range(8):
    #         # data  = tf.string_split(batch,'//t')
    #         # print(data.eval())
    #         print(batch.eval().decode('utf-8'))
    #         # print(batch_1.eval().decode('utf-8'))
    #         print(ids.eval())

    # gen_data()
    #
    # length = len(query)
    # for i in range(length):
    #     print(query[i],answer[i])
    # query_ids=get_word2index(words, query)
    # answer_ids = get_word2index(words, answer)
    # answer = [['芝加哥', '国际', '电影节', 'ChicagoInternationalFilmFestival', '是', '北美', '历史', '最久', '的', '评奖', '电影节'],['最佳', '女', '主角奖', '余', '男', '《', '图雅', '的', '婚事', '》', '中国'],['芝加哥', '电影节', '每年', '10', '月', '举办', '自', '1965', '年', '第一届', '电影节', '至今', '芝加哥', '国际', '电影节', '已', '成为', '世界', '知名', '的', '年度', '电影', '盛会']]
    # test_answer = []
    # for data in answer:
    #     test_answer.append(np.asarray(data))
    # answer_ids = get_word2index(words, np.asarray(test_answer))

