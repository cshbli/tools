import numpy as np
import tensorflow as tf

def save_to_json(model, filename):
    import google.protobuf.json_format as json_format
    json_str = json_format.MessageToJson(model, preserving_proto_field_name = True)

    with open(filename, "w") as of:
        of.write(json_str)

    print ("IR network structure is saved as [{}].".format(filename))

    return json_str


def save_to_proto(model, filename):
    proto_str = model.SerializeToString()
    with open(filename, 'wb') as of:
        of.write(proto_str)

    print ("IR network structure is saved as [{}].".format(filename))

    return proto_str


def load_weights(file_name=None):
    try:
        weights_dict = np.load(file_name).item()
    except:
        weights_dict = np.load(file_name, encoding='bytes').item()
    return weights_dict


def save_weights(weights, filename):
    with open(filename, 'wb') as of:
        np.save(of, weights)
    print ("IR weights are saved as [{}].".format(filename))


def load_graph(frozen_graph_filename):
    """Load a (frozen) Tensorflow model into memory.

    Args:
        frozen_graph_filename:  frozen Tensorflow graph .pb file name

    Returns:
        graph: Tensorflow Graph        
    """
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')
    return graph


def preprocess_image(x, image_format='RGB', mode='caffe'):
    """ Preprocess an image by subtracting the ImageNet mean or divided by 127.5

    Args
        x:              np.array of shape (None, None, 3) or (3, None, None).
        image_format:   "RGB" or "BGR"
        mode:           One of "caffe" or "tf".
                - caffe: will zero-center each color channel with
                         respect to the ImageNet dataset, without scaling.
                - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The normalized image
    """

    # covert to float32
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        if image_format == 'RGB':
            x[..., 0] -= 123.68         # Red
            x[..., 1] -= 116.779        # Green
            x[..., 2] -= 103.939        # Blue
        else:
            x[..., 2] -= 123.68         # Red
            x[..., 1] -= 116.779        # Green
            x[..., 0] -= 103.939        # Blue

    return x


def preprocess_image(x, scale, img_mean, img_std):
    """ Preprocess an image by dividing (img / scale - mean) / std
    
    Returns
        The normalized image
    """
    # covert to float32
    x = x.astype(np.float32)

    for i in range(3):        
        x[..., i]= (x[..., i]/scale - img_mean[i])/img_std[i]
    
    return x


def extract_features(inference_graph, input_image, input_layer, output_layers):
    """ Extract features for different layers given the inference_graph, input_image and input_layer name

    Args:
        inference_graph: The loaded inference graph
        input_image:     The preprocessed image Numpy array
        input_layer:     The input layer name
        output_layers:   The list of output layers

    Returns:
        output_dict:    A dictionary whose key is the output layer name plus ":0",
                        whose value is the corresponding output from that layer
    """

    sess = tf.Session(graph=inference_graph)    
    tensor_dict = {}
    for output in output_layers:
        tensor_name = output + ":0"
        tensor_dict[tensor_name] = inference_graph.get_tensor_by_name(tensor_name)
    image_tensor = inference_graph.get_tensor_by_name(input_layer + ":0")
    output_dict = sess.run(tensor_dict,  feed_dict={image_tensor: np.expand_dims(input_image, axis=0)})

    return output_dict


def extract_features_checkpoint_model(checkpoint_path, input_image, input_layer, output_layers):
    """ Extract features for different layers given the inference_graph, input_image and input_layer name

    Args:
        checkpoint_path: The loaded inference graph
        input_image:     The preprocessed image Numpy array
        input_layer:     The input layer name
        output_layers:   The list of output layers

    Returns:
        output_dict:    A dictionary whose key is the output layer name plus ":0",
                        whose value is the corresponding output from that layer
    """

    sess = tf.Session()
    # First load meta graph and restore weights
    saver = tf.train.import_meta_graph(checkpoint_path + '/' + 'model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

    inference_graph = tf.get_default_graph()

    tensor_dict = {}
    for output in output_layers:
        tensor_name = output + ":0"
        tensor_dict[tensor_name] = inference_graph.get_tensor_by_name(tensor_name)
    image_tensor = inference_graph.get_tensor_by_name(input_layer + ":0")
    output_dict = sess.run(tensor_dict,  feed_dict={image_tensor: np.expand_dims(input_image, axis=0)})

    return output_dict