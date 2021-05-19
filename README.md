# Utilities and Tools

* [Summarize Graph](#summarize-graph)

* [Output tensor values for specified layer](#output_tensor_bin)

* [Visualize Tensorflow .pb or .meta model with Tensorboard](./vis_pb_tensorboard.py)

* [ImageNet to Tensorflow TFRecord](./ImageNet-to-TFrecord/README.md)

* [Convert object detection xml annotations to CSV](./xml-to-csv.py)

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

## Output Tensor Bin

- For frozen graph

```
python output_tensor_bin.py \
  --model frozen_quant_graph.pb \
  --input_node=Placeholder \
  --image_width=512 \
  --image_height=512 \
  --image_mean="0.0,0.0,0.0" \
  --output_node="hm/conv2d_1/Conv2D/bias" \
  --output_bin_path="bst_quant/bins_22371" \
  --input_image=./calibration/000000022371.jpg
```

- For check point model, the "--model" will be the checkpoint model path, the meta graph is named as "model.ckpt.meta"

```
python /data/Projects/tools/output_tensor_bin.py \
  --model=quant \
  --input_node=Placeholder \
  --image_width=512 \
  --image_height=512 \
  --image_mean="0.0,0.0,0.0" \  
  --output_node="hm/conv2d_1/Conv2D/bias" \
  --output_bin_path="bst_quant/bins_22371" \
  --input_image=./calibration/000000022371.jpg
```