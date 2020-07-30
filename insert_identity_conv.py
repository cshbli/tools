import os
import numpy as np
import tensorflow as tf
import argparse
import re

# queue for python3; Queue for python2
try:
  import queue
except ImportError:
  import Queue as queue

def parse_args():
    parser = argparse.ArgumentParser(description = 'Insert identity Conv nodes for quantization')
    
    parser.add_argument('-input_model',         type=str, default=None, help='The input Tensorflow frozen graph model path')
    parser.add_argument('-output_model',        type=str, default=None, help='The output Tensorflow frozen graph model path')
    parser.add_argument('-output_node_names',        type=str, default=None, help='The output node names')

    return parser.parse_args()


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


def _GetContextFromOp(op):
  """Gets the root context name from the op name."""
  context_re = re.search(r'^(.*)/([^/]+)', op.name)
  if context_re:
    return context_re.group(1)
  return ''


def _AddContextToName(context, name):
  """Adds the context to the name if it exists."""
  if not context:
    return name
  return context + '/' + name


def RerouteTensor(t0, t1, can_modify=None):
  """Reroute the end of the tensor t0 to the ends of the tensor t1.

  Args:
    t0: a tf.Tensor.
    t1: a tf.Tensor.
    can_modify: iterable of operations which can be modified. Any operation
      outside within_ops will be left untouched by this function.

  Returns:
    The number of individual modifications made by the function.
  """
  nb_update_inputs = 0
  consumers = t1.consumers()
  if can_modify is not None:
    consumers = [c for c in consumers if c in can_modify]
  consumers_indices = {}
  for c in consumers:
    consumers_indices[c] = [i for i, t in enumerate(c.inputs) if t is t1]
  for c in consumers:
    for i in consumers_indices[c]:
      c._update_input(i, t0)  # pylint: disable=protected-access
      nb_update_inputs += 1
  return nb_update_inputs


def _insert_identity_conv(in_edge_op, op):
    """Insert one Identity Conv for in_edge->op"""

    print("input_op.name: {}, type: {}".format(in_edge_op.name, in_edge_op.type))
    input_tensor_shape = graph.get_tensor_by_name(in_edge_op.name + ":0").shape
    input_tensor_channels = input_tensor_shape[3]
    print("input_tensor_channels: {}".format(input_tensor_channels))

    context = _GetContextFromOp(in_edge_op)
    print("context: {}".format(context))

    # Construct a Identity Convolution Weight
    identity_matrix = np.identity(input_tensor_channels).astype(np.float32)
    weight_matrix = np.reshape(identity_matrix, (1, 1, input_tensor_channels, input_tensor_channels))
    name_prefix = _AddContextToName(context, "IdentityConv")
    # For frozen graph, we have to use tf.constant
    w = tf.constant(weight_matrix, name=name_prefix + "_weight")    

    inputs = in_edge_op.outputs[0]
    identity_conv_op = tf.nn.convolution(inputs, w, strides=[1, 1], padding='SAME', name=name_prefix)

    tensors_modified_count = RerouteTensor(identity_conv_op, inputs, can_modify=[op])
    print("Insert one Identity Conv: {}".format(name_prefix))
    print("")


def bfs(graph, parent, son):
  ''' check if parent node is an ancesotor of the son node.
      Returns 1 if yes, 0 otherwise
  '''
  q = queue.Queue()
  visited = set()

  q.put(graph.get_operation_by_name(son))

  while not q.empty():
    new_q = queue.Queue()

    while not q.empty():
      new_son = q.get()
      if new_son.name == parent:
        return 1
      for in_edge in new_son.inputs:
        ops = in_edge.op
        if ops not in visited:
          # keep track of the visited (edge). Do not add the
          # edge to the queue if it was visited before. This is
          # need to prevent O(2^h), where h is the son node's
          # depth/height
          visited.add(ops)
          new_q.put(ops)
    q = new_q

  return 0


def checkLink(graph, id_pair, node1_name, node2_name):
  """ Check if node1 is the parent node of node2

  Args
    graph:      A Tensorflow Graph
    id_pair:    A dictionary of operation name to id
    node1_name: Op node1 name
    node2_name: Op node2 name  

  Returns
    0:    No link between node1 and node2
    1:    if node1 is one parent of node2 or node2 is one parent of node1    
  """

  node1_id = id_pair[node1_name]
  node2_id = id_pair[node2_name]
  # print("------")
  # print("node 1 id: {}, name: {}".format(node1_id, node1_name))
  # print("node 2 id: {}, name: {}".format(node2_id, node2_name))
  # node1_id < node2_id, check if node1 is a parent of node2
  if node1_id < node2_id:     
    op = graph.get_operation_by_name(node2_name)
    # if node2 has no any inputs, we are sure node1 is not parent of node2
    if len(op.inputs) == 0:
      # print("No inputs for {}".format(node2_name))
      return 0

    # We need check every input branch of node2
    inputs_checked = 0
    for in_edge in op.inputs:
      # node1 is one of inputs of node2
      if in_edge.op.name == node1_name:
        # print("in_edge.op.name: {}".format(in_edge.op.name))
        inputs_checked = inputs_checked + 1
        return 1

      # if node1 id is larger than node2 input id, then this input branch is done
      if id_pair[in_edge.op.name] < node1_id:
        # print("in_edge.op.name: {}, id: {}".format(in_edge.op.name, id_pair[in_edge.op.name]))
        inputs_checked = inputs_checked + 1
      else:
        # recursively check this input branch
        if checkLink(graph, id_pair, node1_name, in_edge.op.name) == 1:
          return 1
        else:
          inputs_checked = inputs_checked + 1
      
      # All input branches have been checked and none of the branches has the node1 as parent
      if inputs_checked == len(op.inputs):
        # print("all inputs checked")
        return 0
  else:
    # node1_id > node2_id, check if node2 is a parent of node1
    return checkLink(graph, id_pair, node2_name, node1_name)


def _FindShortcutPath(graph):
  id_pair = dict()
  i = 0
  for oper in graph.get_operations():
    id_pair[oper.name] = i
    i += 1
  shortcut_pair = dict()
  for oper in graph.get_operations():
    if oper.type == 'Add':      
      inputs = oper.inputs
      # NOTE(chenghaz): workaround for pooling layer in shortcut path
      if inputs[0].op.type == 'MaxPool':
        inputs = [inputs[0].op.inputs[0], inputs[1]]
      if inputs[1].op.type == 'MaxPool':
        inputs = [inputs[0], inputs[1].op.inputs[0]]
      shortcut = 0
      if id_pair[inputs[0].op.name] < id_pair[inputs[1].op.name]:
        shortcut = bfs(graph, inputs[0].op.name, inputs[1].op.name)
        if shortcut == 1:
          shortcut_pair[oper.name] = inputs[0].op.name          
      else:
        shortcut = bfs(graph, inputs[1].op.name, inputs[0].op.name)
        if shortcut == 1:
          shortcut_pair[oper.name] = inputs[1].op.name
  
  return shortcut_pair


if __name__ == '__main__':
    args = parse_args()

    graph = load_graph(args.input_model)

    with graph.as_default():
        # insert Indentity Conv node for Add->Add
        for add_op in [op for op in graph.get_operations()
                        if op.type == 'AddV2' or op.type == 'Add']:
            print("add_op.name: {}".format(add_op.name))            
            for in_edge in add_op.inputs:
                if in_edge.op.type == "AddV2" or in_edge.op.type == "Add":
                    _insert_identity_conv(in_edge.op, add_op)                    

        # insert Indentity Conv for each Concat input
        for concat_op in [op for op in graph.get_operations()
                            if op.type == 'ConcatV2']:
            print("concat_op.name: {}".format(concat_op.name))
            for in_edge in concat_op.inputs:
                if in_edge.op.name.endswith('axis') and in_edge.op.type == 'Const':
                    # print("skip axis input")
                    continue
                else: 
                    _insert_identity_conv(in_edge.op, concat_op)
                    
        shortcut_pairs = _FindShortcutPath(graph)
        print("Shortcut_pairs: ")
        for shortcut in shortcut_pairs:
            print("oper.name: {}, input: {}".format(shortcut, shortcut_pairs[shortcut]))
            input_op = graph.get_operation_by_name(shortcut_pairs[shortcut])
            op = graph.get_operation_by_name(shortcut)
            _insert_identity_conv(input_op, op)
    
    # for op in graph.get_operations():
    #     print(op.name)
    
    sess = tf.Session(graph=graph)
    # Can't do initializing with frozen graph
    # sess.run(tf.global_variables_initializer())
    output_node_names = args.output_node_names
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,   # The session is used to retrieve the weights
        graph.as_graph_def(), # The graph_def is used to retrieve the nodes 
        output_node_names.split(",")    # The output node names are used to select the usefull nodes        
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(args.output_model, "wb") as f:
        f.write(output_graph_def.SerializeToString())
