# encoding: utf-8
"""
@author: gallupliu
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: train.py
@time: 2018/3/5 22:58


"""

import tensorflow as tf
from classify.dataset import data_utils
from sklearn.model_selection import train_test_split

from  classify.model import TextCNN

def dataset_input_fn(ids,labels,batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((ids, labels))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    return dataset

if __name__ == "__main__":


    text,labels = data_utils.loadfile('./data_with_label.csv')
    word2idx, vocab = data_utils.load_embedding('./dataset/test_cut.txt', './dataset/wiki_50.model')
    ids = data_utils.get_sentence_ids(text, word2idx)
    train_ids,test_ids,train_labels,test_labels = train_test_split(ids,labels,test_size=0.1)
    # print(len(text),type(text))
    # max_length = count_length(text)
    # print(max_length)
    # train_word2vec()

    # print(type(text))
    # print(list(word2idx.keys()))


    # dataset =  tf.data.Dataset.from_tensor_slices((ids, train_labels))
    # iterator = dataset.make_initializable_iterator()
    # next_element = iterator.get_next()
    train_dataset = dataset_input_fn(train_ids, train_labels, 100)
    val_dataset = dataset_input_fn(train_ids, train_labels, 100)
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
    next_element,labels = iterator.get_next()
    train_iterator_init_op = iterator.make_initializer(train_dataset)
    val_iterator_init_op = iterator.make_initializer(val_dataset)


    with tf.Session() as sess:
        # sess.run(iterator.initializer)
        # print(sess.run(next_element))
        model = TextCNN(next_element,labels,vocab,120,3,[1,2,3,5],512)
        sess.run(tf.global_variables_initializer())
        # _,acc,loss = sess.run([model.train_op,model.accuracy,model.loss])
        # print(acc,loss)
        for _ in range(10):
            #训练
            sess.run(train_iterator_init_op)
            feed_dict = {model.dropout_keep_prob:1.0}
            while True:
                try:

                    _, acc, loss = sess.run([model.train_op, model.accuracy, model.loss],feed_dict=feed_dict)
                    print(acc,loss)
                    # print(sess.run(next_element),sess.run(labels))
                except tf.errors.OutOfRangeError:
                    break

