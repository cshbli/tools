import argparse
import tensorflow as tf
from tensorflow.summary import FileWriter
from tensorflow.core.framework import graph_pb2


def parse_args():
    parser = argparse.ArgumentParser(description = 'Visualize Tensorflow .pb model with Tensorboard')    

    parser.add_argument('--logdir',     type=str, default="__tb", help="The output logdir")
    parser.add_argument('model_path',   type=str, default = None, help = 'The Tensorflow model path')

    return parser.parse_args()


def main(args):
    sess = tf.Session()
    if args.model_path.endswith('.pb'):         
        with tf.io.gfile.GFile(args.model_path, "rb") as f:
            graph_def = graph_pb2.GraphDef()
            graph_def.ParseFromString(f.read())

            # fix nodes
            # https://github.com/onnx/tensorflow-onnx/issues/77
            # import frozen graph with error "Input 0 of node X was passed float from Y:0 incompatible with expected float_ref."
            for node in graph_def.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']

            # Set name='', otherwise, the default 'import' will be applied
            tf.import_graph_def(graph_def, name='')
    elif args.model_path.endswith('.meta'):
        tf.train.import_meta_graph(args.model_path)

    FileWriter(args.logdir, sess.graph)
    print("Model Imported. Visualize by running: tensorboard --logdir=" + args.logdir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
  
