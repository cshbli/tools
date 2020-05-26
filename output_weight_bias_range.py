import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description = 'output weights and bias range from an IR npy')

    parser.add_argument('--npy',     type=str, default='IR.npy', help="The IR weight numpy file")

    return parser.parse_args()


def main(args):
    data = np.load(args.npy)

    for layer_name in data[()]:
        output = layer_name
        for param_name in data[()][layer_name]:
            # print(param_name)
            if (param_name == 'weights'):
                weight = data[()][layer_name][param_name]
                rmax, rmin = np.max(weight), np.min(weight)
                output = output + ", " + param_name + ": " + "[{:.4f}, {:.4f}]".format(rmin, rmax)
                # print("[{:.4f}, {:.4f}]".format(wmin, wmax))
            elif (param_name == 'bias'):
                bias = data[()][layer_name][param_name]
                rmax, rmin = np.max(bias), np.min(bias)
                output = output + ", " + param_name + ": " + "[{:.4f}, {:.4f}]".format(rmin, rmax)
                # print("[{:.4f}, {:.4f}]".format(bmin, bmax))
        print(output)

if __name__ == "__main__":
    args = parse_args()
    main(args)