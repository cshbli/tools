import time, os, sys
import logging
import argparse
import numpy as np
import math as m

logger = logging.getLogger('mse')

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate error of a quantized tensor.')
    parser.add_argument('-shape', dest='shape', required=True, type =str,    help='shape of the multi-dimensional tensor')
    parser.add_argument('-qtype', dest='qtype', type =str,   default='int8', help='Quantized data type: int8, uint8, int16')
    parser.add_argument('-mean',  dest='mean',  type =float, default=0.5,    help='mean value of the tensor elements')
    parser.add_argument('-std',   dest='std',   type =float, default=2.0,    help='standard deviation of the tensor elements')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return args

#generate tensor with shape, mean and std
def gen_tensor(mean, std, shape):
    #tensor = np.zeros(shape, dtype=float)
    tensor = np.random.normal(mean, std, shape)
    tensor = tensor.astype(np.float32)
    return tensor

def quantize(inp, qtype):
    shape = inp.shape

    vmin =np.min(inp)
    vmax = np.max(inp)
    print('Data ranage min: {} and max: {}'.format(vmin, vmax))

    if qtype == 'int8':
        scale = 127.0/max(np.absolute(vmax), np.absolute(vmin))
        shift = 0.0
        inp = inp*scale
        qd = inp.astype(np.int8)
    elif qtype == 'uint8':
        scale = 255.0/np.absolute(vmax-vmin)
        shift = vmin
        inp = (inp-shift)*scale
        qd = inp.astype(np.uint8)
    elif qtype == 'int16':
        scale = 32767.0/max(np.absolute(vmax), np.absolute(vmin))
        inp = inp*scale
        shift = 0.0
        qd = inp.astype(np.int16)
    else:
        scale = 1.0
        shift = 0.0
        qd=np.zeros(shape, dtype='int8')
        pass

    return scale, shift, qd

def dequantize(qd, scale, shift):
    dout = qd.astype(np.float32)
    dout = dout/scale+shift
    return dout

def calc_err(inp, ref):
    merr = 0
    mse = 0
    rmse = 0
    psnr = 0

    err = inp-ref
    merr = np.amax(np.absolute(err))
    mse = np.mean(err**2)
    rmse = m.sqrt(mse)
    maxi = np.amax(ref)
    psnr = 10*m.log10((maxi**2)/mse)

    return merr, mse, rmse, psnr
    pass

def main():
    #parse command inputs
    args = parse_args()

    shape = args.shape.split(',')
    while len(shape) < 4:
        shape.append('1')
    shape = map(int, shape)

    mean = args.mean
    std = args.std
    print('Generate tensor with mean {} and standard deviation {} of shape {}'.format(mean, std, shape))
    ref = gen_tensor(mean, std, shape)
    print('Original array:', ref)
    pass

    #quantize the original array
    qtype = args.qtype
    scale, shift, qd = quantize(ref, qtype)
    print ('Quant parameters scale: {} shift: {}'.format(scale, shift))
    print ('Quantized array:', qd)

    #dequantize
    dout = dequantize(qd, scale, shift)
    print('Dequantized array:', dout)

    merr, mse, rmse, psnr = calc_err(dout, ref)
    print("max_err=%e,mse=%e,rmse=%e,psnr=%f" % (merr,mse,rmse,psnr)) 

if __name__=='__main__':
    main()

