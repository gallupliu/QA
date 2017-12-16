# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: test.py
@time: 2017/12/16 19:13


"""
from data import Dataqa,QAFactory

class insuraceqa(Dataqa):
    def gen_train(self):
        print("gen insuranceqa!")

    def gen_embeddings(self):
        print("gen vocab insurance")

class insuranceQAFactory(QAFactory):
    def read_data(self):
        return insuraceqa()