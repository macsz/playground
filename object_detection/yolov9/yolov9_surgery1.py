import onnx
import onnxruntime as ort
import numpy as np
import os
from sima_utils.onnx import onnx_helpers as oh
from onnx import shape_inference
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
import torch
import time
import torchvision
import torch.nn.functional as F
import shutil

yolov9s_dseg = "yolov9/yolov9-c-seg.onnx"

m = onnx.load(yolov9s_dseg)

mask_coefs = ["/model.44/cv7.0/cv7.0.2/Conv_output_0",
              "/model.44/cv7.1/cv7.1.2/Conv_output_0",
              "/model.44/cv7.2/cv7.2.2/Conv_output_0"]
cls_logits = ["/model.44/cv5.0/cv5.0.2/Conv_output_0",
              "/model.44/cv5.1/cv5.1.2/Conv_output_0",
              "/model.44/cv5.2/cv5.2.2/Conv_output_0"]
bbox_potentials = ["/model.44/cv4.0/cv4.0.1/act/Mul_output_0",
                   "/model.44/cv4.1/cv4.1.1/act/Mul_output_0",
                   "/model.44/cv4.2/cv4.2.1/act/Mul_output_0"]
mask = ["output"]
inp = ["images"]

yolov9s_dseg_extracted = "yolov9/yolov9-c-seg_extracted.onnx"
yolov9s_dseg_post_surg = "yolov9/yolov9s_dseg_post_surgery.onnx"

onnx.utils.extract_model(yolov9s_dseg, yolov9s_dseg_extracted, inp, bbox_potentials+cls_logits+mask_coefs+mask)
m_extracted = onnx.load(yolov9s_dseg_extracted)

def change_op_name(m, old_op_name, new_op_name):
    for op in m.graph.output:
        if op.name == old_op_name:
            op.name = new_op_name

for i, cls_logit in enumerate(cls_logits):
    cls_logit_node = cls_logit.strip("_output_0")
    n = onnx.helper.make_node("Sigmoid", inputs=[cls_logit],
                               outputs=[f"cls_prob_{i}"])
    oh.insert_node(m_extracted, cls_logit_node, n, insert_only=True)

cls_probs_op = [f"cls_prob_{i}" for i in range(3)]

for old_cls_op, new_op in zip(cls_logits, cls_probs_op):
    change_op_name(m_extracted, old_cls_op, new_op)

onnx.save(m_extracted, "yolov9/yolov9_dseg_edited.onnx")

bbox_convs = ["/model.44/cv4.0/cv4.0.2/Conv",
              "/model.44/cv4.1/cv4.1.2/Conv",
              "/model.44/cv4.2/cv4.2.2/Conv"]

followup_conv = "/model.44/dfl2/conv/Conv"

anchor_points = "/model.44/Constant_7_output_0"
strides = "/model.44/Constant_10_output_0"

anchor_tensor = oh.find_initializer_value(m, anchor_points)
stride_tensor = oh.find_initializer_value(m, strides)

w_template = [[[[-0.5]],[[0]],[[0.5]],[[0]]],
              [[[0]],[[-0.5]],[[0]],[[0.5]]],
              [[[1]],[[0]],[[1]],[[0]]],
              [[[0]],[[1]],[[0]],[[1]]]]
w_template = np.array(w_template).astype(np.float32)
strides = [8, 16, 32]
a1 = anchor_tensor[:,:,:6400].reshape(1,2,80,80)
zs = np.zeros((1,2,80,80)).astype(np.float32)
prep_a1 = np.concatenate((a1, zs), axis=1)
prep_a1 = prep_a1*8

a2 = anchor_tensor[:,:,6400:6400+1600].reshape(1,2,40,40)
zs = np.zeros((1,2,40,40)).astype(np.float32)
prep_a2 = np.concatenate((a2, zs), axis=1)
prep_a2 = prep_a2*16

a3 = anchor_tensor[:,:,6400+1600:].reshape(1,2,20,20)
zs = np.zeros((1,2,20,20)).astype(np.float32)
prep_a3 = np.concatenate((a3, zs), axis=1)
prep_a3 = prep_a3*32
add_tensors = [prep_a1, prep_a2, prep_a3]

collect_nodes = []
collect_inits = []
collect_inps = []
collect_ops = []
inp_shapes = []
op_shapes = []
szs = [80, 40, 20]
for bbox_conv_idx, conv_name in enumerate(bbox_convs):
    node_conv = oh.find_node(m, conv_name)
    conv_extracted = oh.extract_model(m, node_conv.input[0:1], node_conv.output)
    followup_conv_node = oh.find_node(m, followup_conv)
    # build equivalent graph and infer
    # Add a split to feed convs
    # split = onnx.helper.make_node("Split",
    #                         inputs=[f"cust_{node_conv.input[0]}"],
    #                         outputs=[f"bbox_{bbox_conv_idx}_split_op_0",
    #                                  f"bbox_{bbox_conv_idx}_split_op_1",
    #                                  f"bbox_{bbox_conv_idx}_split_op_2",
    #                                  f"bbox_{bbox_conv_idx}_split_op_3"],
    #                         axis=1,
    #                         num_outputs=4)
    # collect_nodes.append(split)
    # add a slice node for each conv
    for i in range(4):
        n = onnx.helper.make_node("Slice", inputs=[f"cust_{node_conv.input[0]}", "starts", "ends", "axes"],
                              outputs=[f"bbox_{bbox_conv_idx}_slice_{i}_op"])
        s = onnx.helper.make_tensor(name=f"bbox_{bbox_conv_idx}_slice_{i}_starts", data_type=onnx.TensorProto.INT32,
                                dims=[1],
                                vals=[16*i])
        e = onnx.helper.make_tensor(name=f"bbox_{bbox_conv_idx}_slice_{i}_ends", data_type=onnx.TensorProto.INT32,
                                dims=[1],
                                vals=[16*(i+1)])
        a = onnx.helper.make_tensor(name=f"bbox_{bbox_conv_idx}_slice_{i}_axes", data_type=onnx.TensorProto.INT32,
                                dims=[1],
                                vals=[1])
        n.input[1] = s.name
        n.input[2] = e.name
        n.input[3] = a.name
        collect_nodes.append(n)
        collect_inits.extend([s,e,a])
    collect_inps.append(f"cust_{node_conv.input[0]}")
    inp_shapes.append([1, 64, szs[bbox_conv_idx], szs[bbox_conv_idx]])
    # add 4 new convs
    new_convs = []
    for i in range(4):
        n = onnx.helper.make_node("Conv", inputs=[f"bbox_{bbox_conv_idx}_slice_{i}_op", "w", "b"],
                              outputs=[f"bbox_{bbox_conv_idx}_conv_{i}_op"], dilations=[1,1],
                              group=1, kernel_shape=[1,1],
                              pads=[0,0,0,0],
                              strides=[1,1])
        new_convs.append(n)
    w = oh.find_initializer_value(conv_extracted, node_conv.input[1])
    b = oh.find_initializer_value(conv_extracted, node_conv.input[2])
    # create weight initializers
    ws = []
    for i in range(4):
        start_idx = 16*i
        wi = w[i*16:(i+1)*16,:,:,:]
        assert wi.shape == (16,16,1,1)
        wi = onnx.helper.make_tensor(name=f"bbox_{bbox_conv_idx}_conv_{i}_w", data_type=onnx.TensorProto.FLOAT,
                                dims=list(wi.shape),
                                vals=wi.flatten().tolist())
        ws.append(wi)
    # create bias initializers
    bs = []
    for i in range(4):
        start_idx = 16*i
        wi = b[i*16:(i+1)*16]
        assert wi.shape == (16,)
        wi = onnx.helper.make_tensor(name=f"bbox_{bbox_conv_idx}_conv_{i}_b", data_type=onnx.TensorProto.FLOAT,
                                dims=list(wi.shape),
                                vals=wi.flatten().tolist())
        bs.append(wi)
    # connect weight and biases with convs
    for i, (conv, w, b) in enumerate(zip(new_convs, ws, bs)):
        conv.input[1] = w.name
        conv.input[2] = b.name
    collect_nodes.extend(new_convs)
    collect_inits.extend(ws)
    collect_inits.extend(bs)
    # add softmax for each new conv
    for i in range(4):
        n = onnx.helper.make_node("Softmax", inputs=[f"bbox_{bbox_conv_idx}_conv_{i}_op",],
                                  outputs=[f"bbox_{bbox_conv_idx}_smax_{i}_op"],
                                  axis=1)
        collect_nodes.append(n)
    # duplicate followup conv after each softmax
    for i in range(4):
        n = onnx.helper.make_node("Conv", inputs=[f"bbox_{bbox_conv_idx}_smax_{i}_op","w"],
                                  outputs=[f"bbox_{bbox_conv_idx}_fup_conv_{i}_op"],
                                  group=1, kernel_shape=[1,1],
                                  pads=[0,0,0,0],
                                  strides=[1,1])
        old_w = oh.find_initializer_value(m, followup_conv_node.input[1])
        #old_b = oh.find_initializer_value(m, followup_conv_node.input[2])
        w = onnx.helper.make_tensor(name=f"bbox_{bbox_conv_idx}_fup_conv_{i}_w", data_type=onnx.TensorProto.FLOAT,
                                dims=list(old_w.shape),
                                vals=old_w.flatten().tolist())
        #b = onnx.helper.make_tensor(name=f"bbox_{bbox_conv_idx}_fup_conv_{i}_b", data_type=onnx.TensorProto.FLOAT,
        #                        dims=list(old_b.shape),
        #                        vals=old_b.flatten().tolist())
        n.input[1] = w.name
        #n.input[2] = b.name
        collect_nodes.append(n)
        collect_inits.append(w)
        #collect_inits.append(b)
    # concat
    n = onnx.helper.make_node("Concat", inputs=[f"bbox_{bbox_conv_idx}_fup_conv_0_op",
                                                f"bbox_{bbox_conv_idx}_fup_conv_1_op",
                                                f"bbox_{bbox_conv_idx}_fup_conv_2_op",
                                                f"bbox_{bbox_conv_idx}_fup_conv_3_op"],
                                  outputs=[f"bbox_{bbox_conv_idx}_concat_op"],
                                  axis = 1)
    collect_nodes.append(n)
    # add a conv to do half of dist2xywh
    n = onnx.helper.make_node("Conv", inputs=[f"bbox_{bbox_conv_idx}_concat_op","w"],
                                  outputs=[f"bbox_{bbox_conv_idx}_dist2xywh.conv_op"],
                                  group=1, kernel_shape=[1,1],
                                  pads=[0,0,0,0],
                                  strides=[1,1])
    w_ndarray = w_template * strides[bbox_conv_idx]
    w = onnx.helper.make_tensor(name=f"bbox_{bbox_conv_idx}_dist2xywh.conv_w", data_type=onnx.TensorProto.FLOAT,
                                dims=list(w_template.shape),
                                vals=w_ndarray.flatten().tolist())
    n.input[1] = w.name
    collect_nodes.append(n)
    collect_inits.append(w)
    # add a Add to do the other half of dist2xywh
    n = onnx.helper.make_node("Add", inputs=[f"bbox_{bbox_conv_idx}_dist2xywh.conv_op","w",],
                                  outputs=[f"bbox_{bbox_conv_idx}_dist2xywh.add_op"])
    w = onnx.helper.make_tensor(name=f"bbox_{bbox_conv_idx}_dist2xywh.add_w", data_type=onnx.TensorProto.FLOAT,
                                dims=list(add_tensors[bbox_conv_idx].shape),
                                vals=add_tensors[bbox_conv_idx].flatten().tolist())
    n.input[1] = w.name
    collect_nodes.append(n)
    collect_inits.append(w)
    collect_ops.append(n.output[0])
    op_shapes.append([1,4,szs[bbox_conv_idx], szs[bbox_conv_idx]])

g = onnx.helper.make_graph(collect_nodes,
                           name="custom_sub_graph",
                           inputs=[onnx.helper.make_tensor_value_info(inp, onnx.TensorProto.FLOAT, shape) for inp, shape in zip(collect_inps, inp_shapes)],
                           outputs=[onnx.helper.make_tensor_value_info(op, onnx.TensorProto.FLOAT, shape) for op, shape in zip(collect_ops, op_shapes)],
                           initializer=collect_inits)

custom_graph = onnx.helper.make_model(g,)
onnx.checker.check_model(custom_graph)
custom_graph_shape = shape_inference.infer_shapes(custom_graph)
onnx.save(custom_graph_shape, "yolov9/custom_graph.onnx")

_ONNX_IR_VERSION = 8
_ONNX_OPSET_VERSION = 17

m.ir_version = _ONNX_IR_VERSION
custom_graph_shape.ir_version = _ONNX_IR_VERSION

m = onnx.version_converter.convert_version(m, _ONNX_OPSET_VERSION)
custom_graph_shape = onnx.version_converter.convert_version(custom_graph_shape, _ONNX_OPSET_VERSION)
yolov9s_pm_edited = None
me = m_extracted
for op in custom_graph_shape.graph.output:
    print(op.name)

io_map = [["/model.44/cv4.0/cv4.0.1/act/Mul_output_0", "cust_/model.44/cv4.0/cv4.0.1/act/Mul_output_0"],
          ["/model.44/cv4.1/cv4.1.1/act/Mul_output_0", "cust_/model.44/cv4.1/cv4.1.1/act/Mul_output_0"],
          ["/model.44/cv4.2/cv4.2.1/act/Mul_output_0", "cust_/model.44/cv4.2/cv4.2.1/act/Mul_output_0"]]

inv_io_map = [[v,k] for k,v in io_map]

me.ir_version = _ONNX_IR_VERSION
me = onnx.version_converter.convert_version(me, _ONNX_OPSET_VERSION)

mm = oh.merge_model(me, custom_graph_shape, io_map)

cm = onnx.compose.merge_graphs(me.graph, custom_graph_shape.graph, io_map,
                               inputs=["images"],
                               outputs=["bbox_0_dist2xywh.add_op",
                                        "bbox_1_dist2xywh.add_op",
                                        "bbox_2_dist2xywh.add_op"]+cls_probs_op+mask_coefs+["output1"])


test_model = onnx.helper.make_model(cm,)

for op in test_model.graph.output:
    print(op.name)

cls_ops = []
bbox_ops = []
mask_op = []
mask_coef_ops = []

for op in test_model.graph.output:
    if "cls_prob" in op.name:
        cls_ops.append(op)
    elif "bbox" in op.name:
        bbox_ops.append(op)
    elif "output1" in op.name:
        mask_op.append(op)
    else:
        mask_coef_ops.append(op)


ops_in_order = bbox_ops+cls_ops+mask_coef_ops+mask_op

for i in range(len(test_model.graph.output)):
    test_model.graph.output.pop(i)

for op in ops_in_order:
    test_model.graph.output.append(op)

test_model.ir_version = _ONNX_IR_VERSION
test_model = onnx.version_converter.convert_version(test_model, _ONNX_OPSET_VERSION)

onnx.save(test_model, "yolov9/yolov9c_dseg_post_surgery.onnx")

oh.save_model(test_model, "yolov9/yolov9c_dseg_post_surgery.onnx")