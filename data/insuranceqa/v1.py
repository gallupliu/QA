# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: v1.py
@time: 2018/1/12 23:06


"""

from data import QAData
from data.insuranceqa.reader.v1_reader import V1Reader

#读取V1数据类
class V1Data(QAData):
    def _get_reader(self):
        return V1Reader(self.config['insuranceqa'], self.lowercased, self.logger)


component = V1Data
