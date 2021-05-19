import pandas as pd
import dataset_prep
import depth_prediction_net
import loss
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input, Activation, Add
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K

TF_FORCE_GPU_ALLOW_GROWTH=True


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
get_loss = loss.get_loss()

# # Clear any previous session.
# K.clear_session()

# save_pb_dir = './saved_model/depth_model'
model_fname = './saved_model/depth_model/weights00000100.h5'
# def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
#     with graph.as_default():
#         graphdef_inf = tf.compat.v1.graph_util.remove_training_nodes(graph.as_graph_def())
#         graphdef_frozen = tf.compat.v1.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
#         graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
#         return graphdef_frozen

# # This line must be executed before loading Keras model.
# tf.compat.v1.keras.backend.set_learning_phase(0) 

# model = load_model(model_fname, custom_objects={'autoencoder_loss': get_loss.autoencoder_loss})

# session = tf.compat.v1.keras.backend.get_session()

# input_names = [t.op.name for t in model.inputs]
# output_names = [t.op.name for t in model.outputs]

# # Prints input and output nodes names, take notes of them.
# print(input_names, output_names)

# frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)

# import tensorflow.contrib.tensorrt as trt

# trt_graph = trt.create_inference_graph(
#     input_graph_def=frozen_graph,
#     outputs=output_names,
#     max_batch_size=1,
#     max_workspace_size_bytes=1 << 25,
#     precision_mode='FP16',
#     minimum_segment_size=50
# )

# graph_io.write_graph(trt_graph, "./saved_model/",
#                      "trt_graph.pb", as_text=False)



def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

model = load_model(model_fname, custom_objects={'autoencoder_loss': get_loss.autoencoder_loss})

frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, './', 'xor.pbtxt', as_text=True)
tf.train.write_graph(frozen_graph, './', 'xor.pb', as_text=False)