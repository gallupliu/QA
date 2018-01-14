# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: v2.py
@time: 2018/1/12 23:07


"""

from data import QAData
from data.insuranceqa.reader.v2_reader import V2Reader

#读取V2数据类
class V2Data(QAData):
    def _get_reader(self):
        return V2Reader(self.config['insuranceqa'], self.lowercased, self.logger)


component = V2Data
