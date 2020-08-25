import pickle
import os
import numpy as np

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Extract one tensor from a pickle file and save it to bin file")
    parser.add_argument('-keyName',     type=str, default=None, help='The key name')
    parser.add_argument('-output_file', type=str, default='output_ts.bin', help='The output bin file name')
    parser.add_argument('-activation_shift',    type=float, help='The activation shift for this tensor')
    parser.add_argument('pickle_file',  type=str, default=None, help='The pickle file path')

    return parser.parse_args()


def main(args):    
    dbfile = open(args.pickle_file, 'rb')
    db = pickle.load(dbfile)
    feature = db[args.keyName]
    print("Feature in float: ", feature)
    dbfile.close()

    feature_int = feature * args.activation_shift * 128
    print("Feature quantized: ", feature_int)
    feature_int = np.int8(feature_int)

    print("Feature quantized: ", feature_int)    

    feature_int = np.transpose(feature_int, [0, 3, 1, 2])
    print("Feature transposed: ", feature_int)

    feature_1d = feature_int.flatten().tobytes()
    with open(args.output_file, 'wb') as output:
        output.write(feature_1d)    
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
  