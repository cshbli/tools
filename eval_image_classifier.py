
import io
import os
import argparse
import csv

import numpy as np
import tensorflow as tf

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None


if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


def parse_args():
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluate Image Classifier')    
    
    parser.add_argument('--model',              help="The model to be evaluated", type=str)
    parser.add_argument('--image-width',        help='Required image width of the model', type=int, default=224)
    parser.add_argument('--image-height',       help='Required image height of the model', type=int, default=224)
    parser.add_argument('--color-mode',         help='Required image color mode of the model', type=str, default='BGR')
    parser.add_argument('--image_mean',         help='The image channel means for preprocessing', type=str, default="103.939, 116.779, 123.68")   # For BGR    
    parser.add_argument('--image_scale',        help="The image channel scales for preprocessing", type=str, default="1.0,1.0,1.0")
    parser.add_argument('--output-node-names',  help="Output node names of the model", type=str)
    parser.add_argument('--input-node-names',   help="Input node names of the model", type=str, default='input_1')
    parser.add_argument('--class-index-csv',    help="Class index CSV file", type=str)

    parser.add_argument('input',        help='Input image DIR /image file name', type=str)  

    return parser.parse_args() 


def load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file.
        grayscale: DEPRECATED use `color_mode="grayscale"`.
        color_mode: The desired image format. One of "grayscale", "rgb", "rgba".
            "grayscale" supports 8-bit images and 32-bit signed integer images.
            Default: "rgb".
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported.
            Default: "nearest".

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `load_img` requires PIL.')
    with open(path, 'rb') as f:
        img = pil_image.open(io.BytesIO(f.read()))
        if color_mode == 'grayscale':
            # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
            # convert it to an 8-bit grayscale image.
            if img.mode not in ('L', 'I;16', 'I'):
                img = img.convert('L')
        elif color_mode == 'rgba':
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
        elif color_mode == 'rgb':
            if img.mode != 'RGB':
                img = img.convert('RGB')
        else:
            raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
        if target_size is not None:
            width_height_tuple = (target_size[1], target_size[0])
            if img.size != width_height_tuple:
                if interpolation not in _PIL_INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(
                            interpolation,
                            ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
                resample = _PIL_INTERPOLATION_METHODS[interpolation]
                img = img.resize(width_height_tuple, resample)
        return img


def list_pictures(directory, ext=('jpg', 'jpeg', 'bmp', 'png', 'ppm', 'tif',
                                  'tiff')):
    """Lists all pictures in a directory, including all subdirectories.

    # Arguments
        directory: string, absolute path to the directory
        ext: tuple of strings or single string, extensions of the pictures

    # Returns
        a list of paths
    """
    ext = tuple('.%s' % e for e in ((ext,) if isinstance(ext, str) else ext))    
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if f.lower().endswith(ext)]
    

def preprocess_image(x, img_means, img_scales):
    """ Preprocess an image by (img - means) / scales
    
    Returns
        The normalized image
    """
    # covert to float32
    x = x.astype(np.float32)

    img_channels = len(img_means)
    for i in range(img_channels):
        x[..., i]= (x[..., i] - img_means[i])/img_scales[i]
    
    return x 


def load_graph(frozen_graph_filename):
    """Load a (frozen) Tensorflow model into memory.

    Args:
        frozen_graph_filename:  frozen Tensorflow graph .pb file name

    Returns:
        graph: Tensorflow Graph        
    """
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')
    return graph    

# ground_truth = {'daisy':0, "dandelion":1, "roses":2, "sunflowers":3, "tulips":4}

def main(args):
    # load class index CSV file
    reader = csv.reader(open(args.class_index_csv, 'r'))
    ground_truth = {}
    for row in reader:
        k, v = row
        ground_truth[v] = k

    # load image or image dir    
    if os.path.isdir(args.input):        
        img_list = list_pictures(args.input)
    else:
        img_list = [args.input]    

    # normalize the image input
    img_channels = 3
    img_mean_strs = args.image_mean.split(',')
    img_means = []
    if len(img_mean_strs) == 3:
        img_channels = 3
    elif len(img_mean_strs) == 1:
        img_channels = 1
    for i in range(img_channels):
        img_means.append(float(img_mean_strs[i]))

    img_scale_strs = args.image_scale.split(',')
    img_scales = []    
    for i in range(img_channels):
        img_scales.append(float(img_scale_strs[i])) 

    inference_graph = load_graph(args.model)

    with tf.compat.v1.Session(graph=inference_graph) as sess:    
        correct = 0
        error = 0
        for i, img_filename in enumerate(img_list):
            print("processing {}".format(img_filename))
            dir_name = os.path.split(img_filename)[0]
            dir_name = os.path.split(dir_name)[1]
            value = int(ground_truth.get(dir_name))
            print("ground truth: {}".format(value))

            image_rgb = load_img(img_filename, target_size=(args.image_width, args.image_height))
            input_image = np.asarray(image_rgb)

            if args.color_mode == 'BGR':
                # 'RGB'->'BGR'
                input_image = input_image[..., ::-1]
            else:
                input_image = input_image
            
            # preprocessing image
            input_image = preprocess_image(input_image, img_means, img_scales)

        
            image_tensor = inference_graph.get_tensor_by_name(args.input_node_names + ":0")

            tensor_dict = {}
            for output_node_name in args.output_node_names.split(','):
                tensor_name = output_node_name + ":0"
                tensor_dict[tensor_name] = inference_graph.get_tensor_by_name(tensor_name)    
    
            output_dict = sess.run(tensor_dict,  feed_dict={image_tensor: np.expand_dims(input_image, axis=0)})
            results = np.squeeze(output_dict.get(args.output_node_names + ":0"))
            # print(results)
            print("predicted: {}".format(np.argmax(results)))            
            if np.argmax(results) == value:
                correct = correct + 1
            else:
                error = error + 1

            print("correct: {}".format(correct))
            print("error: {}".format(error))
    
    print("correct: {}".format(correct))
    print("error: {}".format(error))
    print("accuracy: {}".format(correct/(correct+error)))

if __name__ == "__main__":
    args = parse_args()
    main(args)
