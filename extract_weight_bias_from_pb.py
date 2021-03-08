import os
import argparse
import numpy as np
import tensorflow as tf

from utils import makedirs

def parse_args():
    parser = argparse.ArgumentParser(description='Extrace one weight or bias from an PB file')

    parser.add_argument('--pb',             type=str, help="The pb file path")
    parser.add_argument('--layer_name',     type=str, help='The input layer name')    
    parser.add_argument('-bias',            action="store_true")
    parser.add_argument('-weight',          action="store_true")
    parser.add_argument('--output_suffix',  type=str, default='', help='The suffix append to output node name for output bin file name')
    parser.add_argument('--output_path',    type=str, default="./bins", help="The output binary directory name")

    return parser.parse_args()


def main(args):
    makedirs(args.output_path)

    ## In tensorflow the weights are also stored in constants ops
    ## So to get the values of the weights, you need to run the constant ops
    ## It's a little bit anti-intution, but that's the way they do it

    #construct a GraphDef    
    graph_def = tf.GraphDef()
    with open(args.pb, 'rb') as f:
        graph_def.ParseFromString(f.read())

    #import the GraphDef to the global default Graph
    tf.import_graph_def(graph_def, name='')

    # extract all the constant ops from the Graph
    # and run all the constant ops to get the values (weights) of the constant ops
    constant_values = {}
    with tf.Session() as sess:
        constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
        for constant_op in constant_ops:
            value =  sess.run(constant_op.outputs[0])
            constant_values[constant_op.name] = value

            #In most cases, the type of the value is a numpy.ndarray.
            #So, if you just print it, sometimes many of the values of the array will
            #be replaced by ...
            #But at least you get an array to python object, 
            #you can do what other you want to save it to the format you want

            print(constant_op.name)
            print(value)


if __name__ == "__main__":
    args = parse_args()
    main(args)

  
