import argparse
import numpy as np
import math

def parse_args():
    parser = argparse.ArgumentParser(description ='Calculate MSE of two binary tensors')

    parser.add_argument('--bin1',   type=str, help="The first binary tensor file")
    parser.add_argument('--bin2',   type=str, help="The second binary tensor file")

    return parser.parse_args()


def main(args):
    bin1 = np.fromfile(args.bin1, dtype="float32")
    print(np.shape(bin1))
    vmin = np.min(bin1)
    vmax = np.max(bin1)
    print('Data range min: {} and max: {}'.format(vmin, vmax))
    scale = 127.0/max(np.absolute(vmax), np.absolute(vmin))
    #scale = 1.0
    print('Scale: {}'.format(scale))
    # Normalize the bin1 
    bin1 = bin1 * scale
    # Normalize the bin2
    bin2 = np.fromfile(args.bin2, dtype="float32")
    print(np.shape(bin2))
    bin2 = bin2 * scale

    mse = np.square(np.subtract(bin1, bin2)).mean()
    maxi = np.max(bin2)
    # mini = np.min(bin2)
    # print("maxi: {}, mini: {}".format(maxi, mini))
    psnr = 10*math.log10((maxi**2)/mse)
    print('MSE: {:.4f}, PSNR: {:.4f}'.format(mse, psnr))


if __name__ == "__main__":
    args = parse_args()
    main(args)
  
