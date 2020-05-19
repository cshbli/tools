import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

from utils import load_graph
from utils import preprocess_image
from utils import extract_features
from utils import extract_features_checkpoint_model

def parse_args():
    parser = argparse.ArgumentParser(description = 'dump tensor output to a binary file')

    parser.add_argument('--model',          type=str, help="The inference model path")
    parser.add_argument('--input_node',     type=str, help='The input node name')
    parser.add_argument('--image_width',    type=int, help="The input image width")
    parser.add_argument('--image_height',   type=int, help="The input image height")
    parser.add_argument('--image_scale',    type=float, default=1.0, help="The normalization scale")
    parser.add_argument('--image_format',   type=str, default="RGB", help="The image channel format RGB or BGR")
    parser.add_argument('--image_mean',     type=str, default="123.68,116.78,103.94", help="The image channel means")
    parser.add_argument('--image_std',      type=str, default="1.0,1.0,1.0", help="The image channel stds")
    parser.add_argument('--output_node',    type=str, help='output node name')
    parser.add_argument('--input_image',    type=str, help="The input image path")

    parser.add_argument('--output_bin_file',type=str, default="output_ts.bin", help="The output binary file name")

    return parser.parse_args()


def main(args):
    # load the model
    inference_graph = None
    if args.model.endswith('.pb'): 
        inference_graph = load_graph(args.model)

    # load input image and resize to the graph input
    pil_image = Image.open(args.input_image)
    img_resize = pil_image.resize((args.image_width, args.image_height))
    img_resize = np.array(img_resize)

    # Change the PIL loaded input image to BGR format
    if (args.image_format == "BGR"):
        img_resize = np.transpose(img_resize, [2, 0, 1])

    # normalize the image input
    img_mean_strs = args.image_mean.split(',')
    img_means = []
    if len(img_mean_strs) == 3:
        for i in range(3):
            img_means.append(float(img_mean_strs[i]))    

    img_std_strs = args.image_std.split(',')
    img_stds = []
    if len(img_std_strs) == 3:
        for i in range(3):
            img_stds.append(float(img_std_strs[i]))    
    
    # Preprocess the input image based on normalization parameters
    img_input = preprocess_image(img_resize, args.image_scale, img_means, img_stds)

    # Extract the features
    if inference_graph != None:
        feature_dicts = extract_features(inference_graph, img_input, args.input_node, [args.output_node])
    else:
        feature_dicts = extract_features_checkpoint_model(args.model, img_input, args.input_node, [args.output_node])
    
    for i in feature_dicts:
        print(i)
        features = feature_dicts[i]
        print(np.shape(features))
                
        feature_1d = features[0].flatten().tobytes()
        
        with open(args.output_bin_file, 'wb') as output:
            output.write(feature_1d)


if __name__ == "__main__":
    args = parse_args()
    main(args)
  
