import argparse
import tensorflow as tf
from tensorflow.summary import FileWriter
from tensorflow.core.framework import graph_pb2


def parse_args():
    parser = argparse.ArgumentParser(description = 'Visualize Tensorflow .pb model with Tensorboard')    

    parser.add_argument('model_path', type = str, help = 'The Tensorflow model path', default = None)

    return parser.parse_args()


def main(args):
    sess = tf.Session()
    with tf.io.gfile.GFile(args.model_path, "rb") as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)

    FileWriter("__tb", sess.graph)    
    print("Model Imported. Visualize by running: tensorboard --logdir=__tb")


if __name__ == "__main__":
    args = parse_args()
    main(args)
  
