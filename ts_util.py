
'''
This file offers 3 functions that allow you to compare the output of
two files (either bin or txt). They serve to measure the degree to which
two files differ from each other through Mean Square Error (MSE)

'''

import numpy as np
import math
import os
import re
from sys import argv

import click

def cvt_bin_2_txt(in_bytes, outf):
    """Convert tensor-format binary file to reference model txt format

    Args:
        in_file: input bytes
        out_file: output filename.
    """
    TXT_FMT_DATA_SIZE = 1 # unit: byte
    TXT_FMT_WORD_SIZE = 4
    TXT_FMT_WORD_PER_LINE = 8

    tensor_flatten = in_bytes
    data_per_line = int(TXT_FMT_WORD_PER_LINE * TXT_FMT_WORD_SIZE /
                        TXT_FMT_DATA_SIZE)
    num_line = int(math.ceil(len(tensor_flatten) / data_per_line))
    # len of data on last line
    data_at_last_line = len(tensor_flatten) - (num_line - 1) * data_per_line

    for i in range(num_line):
        # if last line
        if i == num_line - 1:
            for j in range(int((data_per_line - data_at_last_line) /
                               TXT_FMT_WORD_SIZE)):
                outf.write('xxxxxxxx ')
            remaining_x_cnt = (data_per_line - data_at_last_line) % TXT_FMT_WORD_SIZE
            if remaining_x_cnt != 0:
                for j in range(remaining_x_cnt):
                    outf.write('xx')
                for j in range(TXT_FMT_WORD_SIZE - remaining_x_cnt):
                    outf.write('{0:02x}'.format(tensor_flatten[-(j+1)]))
                outf.write(' ')
            # Write actual content
            for j in np.arange(int(data_at_last_line / TXT_FMT_WORD_SIZE),
                               0, -1):
                idx = i * data_per_line + j * TXT_FMT_WORD_SIZE
                for k in range(TXT_FMT_WORD_SIZE):
                    outf.write('{0:02x}'.format(tensor_flatten[idx - k - 1]))
                outf.write(' ')
            outf.write('@00000000\n')
        # if not last line
        else:
            for j in np.arange(TXT_FMT_WORD_PER_LINE, 0, -1):
                idx = i * data_per_line + j * TXT_FMT_WORD_SIZE
                for k in range(TXT_FMT_WORD_SIZE):
                    outf.write('{0:02x}'.format(tensor_flatten[idx - k - 1]))
                outf.write(' ')
            outf.write('@00000000\n')

def cvt_txt_2_byte(in_file, result_holder):
    '''
    Convert txt into bytearray
    '''
    def _line_2_bin(line_str):
        # Strip the padding x's
        line_str = line_str.lstrip('x')
        # Convert into bytes
        out = bytearray.fromhex(line_str)
        # Reserve byte-wise
        out = out[::-1]

        return out
    has_dwh = False
    with open(in_file, 'r') as fin:
        # Read every line except the last line
        for line in fin.readlines():
            # Convert line into a single string with no space
            if line.startswith('#'):
                m = re.search(r'w=(\d+) h=(\d+) d=(\d+)',
                              line)
                if m:
                    has_dwh = True
                    w = int(m.group(1))
                    h = int(m.group(2))
                    d = int(m.group(3))
                continue
            line = line.rstrip()
            list_of_words = line.split()[:-1] # Remove the '@00000000' at the end
            line_str = ''.join(list_of_words)
            # Convert string into byte
            line_byte = _line_2_bin(line_str)
            result_holder.extend(line_byte)

    if has_dwh:
        # Note: this only works with single call of
        # the current method.
        print("reshape tensor ", w, h, d)
        ts = np.array(result_holder).reshape(d, h, w)
        print(bytes(ts[0:10, 0, 0]).hex())
        result_holder = ts.transpose(1, 2, 0).flatten()
        print(bytes(result_holder[0:10]).hex())
    return result_holder

def concat_txt_2_bytes(base_dir, file_list):
    ''' concat multiple .txt files into a byte array '''
    result = bytearray()
    for f in file_list:
        filepath = os.path.join(base_dir, f)
        cvt_txt_2_byte(filepath, result)
    return result

def file_to_array(path):
    '''
    Read file (bin or txt) and convert to np.uint8 array
    '''
    _, extension = os.path.splitext(path)
    if extension == '.bin':
        with open(path, 'rb') as fin:
            tensor_flatten = np.frombuffer(fin.read(), dtype='int8')
            return tensor_flatten
    elif extension == '.txt':
        result = bytearray()
        cvt_txt_2_byte(path, result)
        tensor_flatten = np.frombuffer(result, dtype='int8')
        return tensor_flatten
    raise ValueError('Unknown file extension, must be ".bin" or ".txt"')

def compute_mse(file1, file2):
    # Read in files
    arr1 = file_to_array(file1)
    arr2 = file_to_array(file2)
    arr1_len, arr2_len = len(arr1), len(arr2)
    # Check length
    if arr1_len != arr2_len:
        print(f'Unequal lengths, File 1 has length {arr1_len} while File 2 has length {arr2_len}')
        return (np.inf, np.inf)

    mse = (np.square(arr1 - arr2)).mean()
    mae = (abs(arr1 - arr2)).mean()
    return (mse, mae)

@click.command()
@click.option('--file1', type=click.Path(exists=True), required=False,
              help='file path1 for comparison')
@click.option('--file2', type=click.Path(exists=True), required=False,
              help='file path2 for comparison')
@click.option('--txtfile', required=False,
              help='.txt file for binary conversion')
@click.option('--binfile', required=False,
              help='.bin file for txt conversion')
@click.option('--outfile', help='output file path',  required=False,)
@click.option('--dwh', default='', help='tensor size in d,w,h format')
@click.option('--tf', default=False, is_flag=True,
              help='tensor size in d,w,h format')
def cli(file1, file2, txtfile, binfile, outfile,
        dwh, tf):
    if file1 and  file2:
        mse, mae = compute_mse(file1, file2)
        print(f"MSE={mse}, MAE={mae}")
        return

    if txtfile:
        result = bytearray()
        result = cvt_txt_2_byte(txtfile, result)
        print(bytes(result[0:10]).hex())
        with open(outfile, "wb") as fout:
          fout.write(result)
        return
    
    if binfile:
        with open(binfile, 'rb') as f_bin:
            in_bytes = np.frombuffer(f_bin.read(),
                                     dtype='uint8')

            print(bytes(in_bytes[0:10]).hex())
            if not dwh =='':
                ds,ws,hs=dwh.split(',')
                d=int(ds)
                w=int(ws)
                h=int(hs)
                tfbin = in_bytes.reshape(h,w,d)
                # in_bytes=tfbin.transpose(1, 2, 0).flatten()
                tfbin = tfbin.transpose(2,0,1)
                in_bytes=tfbin.flatten()
                print(bytes(in_bytes[0:10]).hex())
            with open(outfile, 'w') as fout:
                cvt_bin_2_txt(in_bytes, fout)

if __name__ == '__main__':
    cli()