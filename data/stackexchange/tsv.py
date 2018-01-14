# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: tsv.py
@time: 2018/1/12 21:43


"""


from data import QAData
from data.stackexchange.reader.tsv_reader import TSVReader

class TSVData(QAData):
    def _get_reader(self):
        return TSVReader(self.config['stackexchange'], self.lowercased, self.logger)