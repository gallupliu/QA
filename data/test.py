# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: test.py
@time: 2017/12/16 19:17


"""
from insuranceqa.test import insuranceQAFactory
from wikiqa.test import wikiQAFactory



if __name__ == '__main__':
    myFactory_1 = wikiQAFactory()
    myFactory_2 = insuranceQAFactory()

    read_data_1 = myFactory_1.read_data()
    read_data_2 = myFactory_2.read_data()

    read_data_1.gen_train()
    read_data_2.gen_train()
