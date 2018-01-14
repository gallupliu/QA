# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: test.py
@time: 2017/12/16 19:17


"""
# from insuranceqa.test import insuraceqa
# from wikiqa.test import wikiqa

#抽象类
class DataSuper(object):
    def __init__(self):
        pass

    def gen_train(self):
        pass

    def gen_embeddings(self):
        pass

#具体策略类
class Data(object):

    def __init__(self,csuper):
        self.csuper = csuper

    def get_data(self):
        return self.csuper.gen_train()


