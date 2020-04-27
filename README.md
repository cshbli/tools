# Utilities and Tools
* [Summarize Graph](#summarize-graph)

* [Visualize Tensorflow .pb or .meta model with Tensorboard](./vis_pb_tensorboard.py)

* [ImageNet to Tensorflow TFRecord](./ImageNet-to-TFrecord/README.md)

## Summarize graph

In order to remove the TensorFlow source build dependency, the independent [Summarize graph tool](tools/summarize_graph.py) is provided to dump the possible inputs nodes and outputs nodes of the graph. It could be taken as the reference list for INPUT_NODE_LIST and OUTPUT_NODE_LIST parameters
of graph_converter class. 

- If use graph in binary,

```
$ python summarize_graph.py --in_graph=path/to/graph --input_binary
```

- Or use graph in text,

```
$ python summarize_graph.py --in_graph=path/to/graph
```

