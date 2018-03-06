# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: test_dataset.py
@time: 2018/2/20 20:02


"""

# import tensorflow as tf
#
# # sequences = [[1, 2, 3], [4, 5, 1], [1, 2]]
# sequences = [["1", "2", "3"], ["4", "5", "1"], ["1", "2"]]
# label_sequences = [[0, 1, 0], [1, 0, 0], [1, 1]]
#
#
# def make_example(sequence, labels):
#     # The object we return
#     ex = tf.train.SequenceExample()
#     # A non-sequential feature of our example
#     sequence_length = len(sequence)
#     ex.context.feature["length"].int64_list.value.append(sequence_length)
#     # Feature lists for the two sequential features of our example
#     fl_tokens = ex.feature_lists.feature_list["tokens"]
#     fl_labels = ex.feature_lists.feature_list["labels"]
#     for token, label in zip(sequence, labels):
#         # fl_tokens.feature.add().int64_list.value.append(token)
#         fl_tokens.feature.add().bytes_list.value.append(token)
#         fl_labels.feature.add().int64_list.value.append(label)
#     return ex
#
#
# # Write all examples into a TFRecords file
# # with tempfile.NamedTemporaryFile() as fp:
# #     writer = tf.python_io.TFRecordWriter(fp.name)
# #     for sequence, label_sequence in zip(sequences, label_sequences):
# #         ex = make_example(sequence, label_sequence)
# #         writer.write(ex.SerializeToString())
# #     writer.close()
#
# with  tf.python_io.TFRecordWriter('./test_0220.tfrecord') as writer:
#     for sequence, label_sequence in zip(sequences, label_sequences):
#         ex = make_example(sequence, label_sequence)
#         writer.write(ex.SerializeToString())
#
#
# # A single serialized example
# # (You can read this from a file using TFRecordReader)
# # ex = make_example([1, 2, 3], [0, 1, 0]).SerializeToString()
#
# ex = make_example(["1", "2", "3"], ["0", "1", "0"]).SerializeToString()
# # Define how to parse the example
# context_features = {
#     "length": tf.FixedLenFeature([], dtype=tf.int64)
# }
# sequence_features = {
#     # "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
#     "tokens": tf.FixedLenSequenceFeature([], dtype=tf.string),
#     "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
# }
#
# # Parse the example
# context_parsed, sequence_parsed = tf.parse_single_sequence_example(
#     serialized=ex,
#     context_features=context_features,
#     sequence_features=sequence_features
# )

import tensorflow as tf
import os
keys=[[1.0,2.0],[2.0,3.0]]
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

def make_example(locale,age,score,times):

    example = tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
            "locale":tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(locale,encoding='utf8')])),
            "age":tf.train.Feature(int64_list=tf.train.Int64List(value=[age]))
        }),
        feature_lists=tf.train.FeatureLists(
            feature_list={
            "movie_rating":tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=score)) for i in range(times)])
            }
        )
    )
    return example.SerializeToString()

context_features = {
    "locale": tf.FixedLenFeature([],dtype=tf.string),
    "age": tf.FixedLenFeature([],dtype=tf.int64)
}
sequence_features = {
    "movie_rating": tf.FixedLenSequenceFeature([3], dtype=tf.float32,allow_missing=True)
}

context_parsed, sequence_parsed  = tf.parse_single_sequence_example(make_example("中国",24,[1.0,3.5,4.0],2),context_features=context_features,sequence_features=sequence_features)

print(tf.contrib.learn.run_n(context_parsed))
print(tf.contrib.learn.run_n(sequence_parsed))