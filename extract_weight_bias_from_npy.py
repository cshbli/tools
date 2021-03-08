import os
import argparse
import numpy as np

from utils import makedirs

def parse_args():
    parser = argparse.ArgumentParser(description='Extrace one weight or bias from a .npy file')

    parser.add_argument('--npy_filename',   type=str, help="The npy file path")
    parser.add_argument('--layer_name',     type=str, help='The input layer name')    
    parser.add_argument('-bias',            action="store_true")
    parser.add_argument('-weights',          action="store_true")
    parser.add_argument('--output_suffix',  type=str, default='', help='The suffix append to output node name for output bin file name')
    parser.add_argument('--output_path',    type=str, default="./bins", help="The output binary directory name")

    return parser.parse_args()


def main(args):
    makedirs(args.output_path)

    # load the npy file
    data = np.load(args.npy_filename)
    for layer_name in data[()]:
        print(layer_name)

    if args.bias:
        bias = data[()][args.layer_name]["bias"]
        print(bias)

        output_filename = args.layer_name.replace('/', '_') + '_bias' + args.output_suffix + '.bin'
        output_filename = os.path.join(args.output_path, output_filename)
        
        bias_1d = bias.flatten().tobytes()
        with open(output_filename, "wb") as output:
            output.write(bias_1d)        

        output_filename = args.layer_name.replace('/', '_') + '_bias' + args.output_suffix + '.txt'
        output_filename = os.path.join(args.output_path, output_filename)

        np.savetxt(output_filename, bias, delimiter=',', fmt="%.6f")

    if args.weights:
        weights = data[()][args.layer_name]["weights"]
        print(weights)

        output_filename = args.layer_name.replace('/', '_') + '_weights' + args.output_suffix + '.bin'
        output_filename = os.path.join(args.output_path, output_filename)
        
        weights_1d = weights.flatten().tobytes()
        with open(output_filename, "wb") as output:
            output.write(weights_1d)        

        output_filename = args.layer_name.replace('/', '_') + '_weights' + args.output_suffix + '.txt'
        output_filename = os.path.join(args.output_path, output_filename)

        weights_1d = weights.flatten()
        np.savetxt(output_filename, weights_1d, delimiter=',', fmt="%.6f")



if __name__ == "__main__":
    args = parse_args()
    main(args)
  
