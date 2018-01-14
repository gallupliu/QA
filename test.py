# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: test.py
@time: 2018/1/12 22:00


"""

from data.test import Data
from data.wikiqa.test import wikiqa
from data.insuranceqa.test import insuraceqa

data1 = Data(wikiqa())
data1.get_data()

data2 = Data(insuraceqa())
data2.get_data()


import os
import importlib
import logging
import sys
from data.config import load_config
# from data.insuranceqa import v2_reader

print("os.path.realpath(__file__)=%s" % os.path.realpath(__file__))
# config = load_config('../configs/insurance/example-config.yaml')
config = load_config('E:\QA\configs\insurance\example-config.yaml')
config_global = config['global']

# setup a logger
logger = logging.getLogger('experiment')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler_stdout = logging.StreamHandler(sys.stdout)
handler_stdout.setLevel(config['logger']['level'])
handler_stdout.setFormatter(formatter)
logger.addHandler(handler_stdout)

if 'path' in config['logger']:
    handler_file = logging.FileHandler(config['logger']['path'])
    handler_file.setLevel(config['logger']['level'])
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)

logger.setLevel(config['logger']['level'])

data_module = config['data-module']
DataClass = importlib.import_module(data_module).component
# We then wire together all the modules and start training
data = DataClass(config['data'], config_global, logger)
# setup the data (validate, create generators, load data, or else)
logger.info('Setting up the data')
data.setup()
logger.info("data loeded")
# data = v2_reader.V2Reader(r'E:\nlp\data\insuranceQA',True,logger)
# data.setup()