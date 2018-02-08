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
import jieba
import numpy as np
import tensorflow as tf


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
    words_list = [''.join(word) for word in jieba.cut(context)]
    return words_list

def get_words_dict():
    reader = open('./test','r',encoding='utf-8')
    lines = reader.readlines()
    words_dict = set()
    query = []
    answer = []
    for line in lines:
        data = line.split('\\t')
        query.append(clean_string(data[0]))
        answer.append(clean_string(data[1]))
        for word in clean_string(line):
            words_dict.add(word)

    words = []
    for word in words_dict:

        words.append(word)
    print(words)
    return np.asarray(words),np.asarray(query),np.asarray(answer)



def get_word2index(mapping,sentence):
    """
    对分词后的list进行word2vec映射
    :param mapping:
    :param sentence:
    :return:
    """
    mapping_strings = tf.constant(mapping)
    sentence_tensor = tf.constant(sentence)
    ids = tf.contrib.lookup.string_to_index(sentence_tensor,mapping_strings,default_value=-1)

    with tf.Session() as sess:
        tf.tables_initializer().run()
        print(sess.run(ids))
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


def read_text_line(filename):
    """
    直接读取原始数据生成dataset
    :param filename:
    :return:
    """

def read_tfrecord():
    """
    直接读取tfreocord生成dataset
    :return:
    """
    pass

if __name__ == "__main__":
    # filenames = ['./test']
    # dataset = tf.data.TextLineDataset(filenames=filenames)
    # #一次只生成一条数据
    # iterator = dataset.make_one_shot_iterator()
    # iterator_1 = dataset.make_initializable_iterator()
    #
    # batch = iterator.get_next()
    # batch_1 = iterator_1.get_next()
    #
    # with tf.Session() as sess:
    #     sess.run(iterator_1.initializer)
    #     for i  in range(8):
    #         print(batch.eval().decode('utf-8'))
    #         print(batch_1.eval().decode('utf-8'))
    #
    # gen_data()
    words, query, answer = get_words_dict()
    # query_ids=get_word2index(words, query)
    answer_ids = get_word2index(words, answer)
