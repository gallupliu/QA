# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: model.py
@time: 2017/12/30 13:23


"""



import tensorflow as tf
from  models import model_utils
from utils.utils import feature2cos_sim, max_pooling, cal_loss_and_acc


class Model():
    def __init__(self, config,embedding):
        self.config = config
        self.embeddings = embedding



    def add_placeholder(self):
        """
        :功能： define input variable
        :return:
        """


    def add_embedding(self):
        """
        :功能：对输入建立索引
        :return:
        """


    def build(self,ids):
        """
        :param ids:
        :return:
        """


    def train_op(self,loss):
        """
        :param loss:
        :return:
        """

