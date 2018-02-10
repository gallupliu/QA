# # encoding: utf-8
# """
# @author: gallupliu
# @contact: gallup-liu@hotmail.com
#
# @version: 1.0
# @license: Apache Licence
# @file: test_variable.py
# @time: 2018/2/10 21:25
#
#
# """
import tensorflow as tf
import numpy as np
import jieba
import re

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
        query.append(clean_string(data[0]))
        answer.append(clean_string(data[1]))
        for word in clean_string(line):
            words_dict.add(word)

    words = []
    for word in words_dict:

        words.append(word)
    return np.asarray(words),np.asarray(query),np.asarray(answer)

def writedata():
    xlist = [[b'1',b'2',b'3'],[b'4',b'5',b'6',b'8']]
    #xlist = [['芝加哥','国际','电影节','最佳','女主角', '是', '谁'], ['2002', '年', '来自', '30', '多个', '国家', '和', '地区', '的', '90', '多部', '故事片', '40', '多部', '短片', '参加', '了', '电影节', '吸引', '了', '世界各地', '的', '6', '万多', '观众']]
    ylist = [1,2]
    #这里的数据只是举个例子来说明样本的文本长度不一样，第一个样本3个词标签1，第二个样本4个词标签2
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for i in range(2):
        x = xlist[i]
        y = ylist[i]
        example = tf.train.Example(features=tf.train.Features(feature={
            "y": tf.train.Feature(int64_list=tf.train.Int64List(value=[y])),
            'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=x))
        }))
        writer.write(example.SerializeToString())
    writer.close()


feature_names = ['x']


def my_input_fn(file_path, mapping_strings,perform_shuffle=False, repeat_count=1):
    def parse(example_proto):
        features = {"x": tf.VarLenFeature(tf.string),
                    "y": tf.FixedLenFeature([1], tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)
        #x = tf.sparse_tensor_to_dense(parsed_features["x"])
        x = parsed_features["x"]
        x = tf.cast(x, tf.string)
        x = tf.contrib.lookup.string_to_index(x,mapping=mapping_strings,default_value=-1)
        x = dict(zip(feature_names, [x]))
        y = tf.cast(parsed_features["y"], tf.int32)
        return x, y

    dataset = (tf.contrib.data.TFRecordDataset(file_path)
               .map(parse))
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.padded_batch(2, padded_shapes=({'x': [6]}, [1]))  # batch size为2，并且x按maxlen=6来做padding
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

words, query, answer = get_words_dict()
# next_batch = my_input_fn('train.tfrecords', words,True)
# init = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(1):
#         xs, y = sess.run(next_batch)
#         print(xs['x'])
#         print(y)


# def writedata():
#     xlist = [[1,2,3],[4,5,6,8]]
#     ylist = [1,2]
#     #这里的数据只是举个例子来说明样本的文本长度不一样，第一个样本3个词标签1，第二个样本4个词标签2
#     writer = tf.python_io.TFRecordWriter("./train.tfrecords")
#     for i in range(2):
#         x = xlist[i]
#         y = ylist[i]
#         example = tf.train.Example(features=tf.train.Features(feature={
#             "y": tf.train.Feature(int64_list=tf.train.Int64List(value=[y])),
#             'x': tf.train.Feature(int64_list=tf.train.Int64List(value=x))
#         }))
#         writer.write(example.SerializeToString())
#     writer.close()
#
# feature_names = ['x']
#
#
# def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
#     def parse(example_proto):
#         features = {"x": tf.VarLenFeature(tf.int64),
#                     "y": tf.FixedLenFeature([1], tf.int64)}
#         parsed_features = tf.parse_single_example(example_proto, features)
#         x = tf.sparse_tensor_to_dense(parsed_features["x"])
#         x = tf.cast(x, tf.int32)
#         x = dict(zip(feature_names, [x]))
#         y = tf.cast(parsed_features["y"], tf.int32)
#         return x, y
#
#     dataset = (tf.contrib.data.TFRecordDataset(file_path)
#                .map(parse))
#     if perform_shuffle:
#         dataset = dataset.shuffle(buffer_size=256)
#     dataset = dataset.repeat(repeat_count)
#     dataset = dataset.padded_batch(2, padded_shapes=({'x': [6]}, [1]))  # batch size为2，并且x按maxlen=6来做padding
#     iterator = dataset.make_one_shot_iterator()
#     batch_features, batch_labels = iterator.get_next()
#     return batch_features, batch_labels
#
writedata()
next_batch = my_input_fn('./train.tfrecords', True)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    tf.tables_initializer().run()
    for i in range(1):
        xs, y = sess.run(next_batch)
        print(xs['x'])
        print(y)


# def load_tfrecord_variable(serialized_example):
#
#     context_features = {
#         'length':tf.FixedLenFeature([],dtype=tf.int64),
#         'batch_size':tf.FixedLenFeature([],dtype=tf.int64),
#         'type':tf.FixedLenFeature([],dtype=tf.string)
#     }
#
#     sequence_features = {
#         "values":tf.VarLenFeature(tf.int64)
#     }
#
#     context_parsed, sequence_parsed = tf.parse_single_sequence_example(
#         serialized=serialized_example,
#         context_features=context_features,
#         sequence_features=sequence_features
#     )
#
#     length = context_parsed['length']
#     batch_size = context_parsed['batch_size']
#     type = context_parsed['type']
#
#     values = sequence_parsed['values'].values
#
#     return tf.tuple([length, batch_size, type, values])
#
# #
# filenames = [fp.name]
#
# dataset = tf.data.TFRecordDataset(filenames)
# dataset = dataset.map(load_tfrecord_fixed)
# dataset = dataset.repeat()
# dataset = dataset.padded_batch(
#     batch_size,
#     padded_shapes=(
#         tf.TensorShape([]),
#         tf.TensorShape([]),
#         tf.TensorShape([]),
#         tf.TensorShape([None])  # if you reshape 'values' in load_tfrecord_variable, add the added dims after None, e.g. [None, 3]
#         ),
#     padding_values = (
#         tf.constant(0, dtype=tf.int64),
#         tf.constant(0, dtype=tf.int64),
#         tf.constant(""),
#         tf.constant(0, dtype=tf.int64)
#         )
#     )
#
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()
#
# with tf.Session() as sess:
#     a = sess.run(iterator.initializer)
#     for i in range(3):
#         [length_vals, batch_size_vals, type_vals, values_vals] = sess.run(next_element)