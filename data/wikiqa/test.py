# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: test.py
@time: 2017/12/16 19:15


"""
from data import Dataqa,QAFactory

class wikiqa(Dataqa):
    def gen_train(self):
        print("gen wikiqa!")

    def gen_embeddings(self):
        print("gen vocab")


class wikiQAFactory(QAFactory):
    def read_data(self):
        return wikiqa()

