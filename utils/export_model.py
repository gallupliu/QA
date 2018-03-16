# encoding: utf-8
"""
@author: gallupliu 
@contact: gallup-liu@hotmail.com

@version: 1.0
@license: Apache Licence
@file: export_model.py
@time: 2018/1/29 22:19


"""
import re
import os
import shutil
import tensorflow as tf

from tensorflow.python.util import compat
from tensorflow.python.client import session
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils


def export_model(input_parameter,output_parameter,sess,export_base_path,version=1.0,use_replice=False):
    #step 0: check the type of input parameter and outprameter
    if not isinstance(input_parameter,dict):
        print("the type of \" input_parameter\" must be dict!")
        return False
    if not isinstance(output_parameter, dict):
        print("the type of \" out_parameter\" must be dict!")
        return False

    if not isinstance(sess, session):
        print("the type of \" sess \" must be dict!")
        return False

    if not isinstance(export_base_path, str):
        print("the type of \" export_base_path \" must be dict!")
        return False
    if not re.match("^[0-9A-Za-z_.\-/]+$",export_base_path):
        print("export_base_path must be presented character-set")
        return False
    export_base_path = os.path.realpath(export_base_path)
    if not isinstance(version, float):
        print("the type of \" verson \" must be float!")
        return False
    if not isinstance(use_replice, bool):
        print("the type of \" use_replice \" must be bool!")
        return False

    export_path = os.path.join(
        compat.as_bytes(export_base_path),
        compat.as_bytes(str(version))
    )

    if os.path.isdir(export_path):
        print("remove the old export_path:%s"%export_path)
        shutil.rmtree(export_path)

    print("export_base_path:"+export_base_path+":model_version:"+str(version))

    if use_replice:
        sess.graph._unsafe_unfinalize()

    #step 2:obtain tensor info of the input placeholder
    input_dict = {}
    output_dict = {}
    for par_name,placehoder in input_parameter.items():
        if not re.match("^[0-9A-Za-z_.\-/]+$",par_name):
            return False
        if not isinstance(placehoder,tf.Tensor):
            return  False
        input_dict[par_name] = utils.build_tensor_info(placehoder)

    for par_name,placehoder in output_parameter.items():
        if not re.match("^[0-9A-Za-z_.\-/]+$",par_name):
            return False
        if not isinstance(placehoder,tf.Tensor):
            return  False
        output_dict[par_name] = utils.build_tensor_info(placehoder)

    prediction_signature = signature_def_utils.build_signature_def(
        inputs=input_dict,
        outputs=output_dict,
        method_name=signature_constants.PREDICT_METHOD_NAME
    )

    #step 3 define the graph of sess
    legacy_init_op = tf.group(tf.tables_initializer(),name="legacy_init_op")

    #step 4 save model
    save_model_builder = builder.SavedModelBuilder(export_path)
    if use_replice:
        save_model_builder.add_meta_graph_and_variables(
            sess,
            [tag_constants.SERVING],
            signature_def_map={
                'predict':prediction_signature
            },
            clear_devices=True,
            legacy_init_op=legacy_init_op
        )
        sess.graph.finalize()
    else:
        save_model_builder.add_meta_graph_and_variables(
            sess,
            [tag_constants.SERVING],
            signature_def_map={
                'prediction':prediction_signature
            },
            legacy_init_op=legacy_init_op
        )
    save_model_builder.save()
    return True



