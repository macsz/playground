#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
#**************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are 
# proprietary to SiMa and may be covered by U.S. and Foreign Patents, 
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is 
# strictly forbidden unless prior written permission is obtained from 
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who 
# have executed Confidentiality and Non-disclosure agreements explicitly 
# covering such access.
#
# The copyright notice above does not evidence any actual or intended 
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE 
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO 
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.                
#
#**************************************************************************
import numpy as np

from sima_utils.onnx import onnx_helpers as oh

# Ultralytics, 640x640, 1-class detection
H, W = 640, 640
num_class = 1
is_segmentation = False
model_name = "yolov8_static_input_v4"

# BBOX OPT rewrite
# Version 1: 4D rewrite
# Version 2: Conv for final element-wise layers
bbox_opt_version = 2
mod_model_name = f"{model_name}_mod_v{bbox_opt_version}"


model_dir = ".."
model = oh.load_model(f"{model_dir}/{model_name}.onnx")

# Remove all outputs and reconstruct outputs.
oh.remove_output(model)
oh.add_output(model, "bbox_0", (1, 4, H//8, W//8))
oh.add_output(model, "bbox_1", (1, 4, H//16, W//16))
oh.add_output(model, "bbox_2", (1, 4, H//32, W//32))
oh.add_output(model, "class_prob_0", (1, num_class, H//8, W//8))
oh.add_output(model, "class_prob_1", (1, num_class, H//16, W//16))
oh.add_output(model, "class_prob_2", (1, num_class, H//32, W//32))

if is_segmentation:
    oh.add_output(model, "mask_coeff_0", (1, 32, H//8, W//8))
    oh.add_output(model, "mask_coeff_1", (1, 32, H//16, W//16))
    oh.add_output(model, "mask_coeff_2", (1, 32, H//32, W//32))
    oh.add_output(model, "mask", (1, 32, H//4, W//4))

    # Modify mask path.
    oh.change_node_output(model, "/model.22/proto/cv3/act/Mul", "mask")

    # Modify mask coeff path.
    oh.change_node_output(model, "/model.22/cv4.0/cv4.0.2/Conv", "mask_coeff_0")
    oh.change_node_output(model, "/model.22/cv4.1/cv4.1.2/Conv", "mask_coeff_1")
    oh.change_node_output(model, "/model.22/cv4.2/cv4.2.2/Conv", "mask_coeff_2")
    oh.remove_node(model, "/model.22/Reshape", True)
    oh.remove_node(model, "/model.22/Reshape_1", True)
    oh.remove_node(model, "/model.22/Reshape_2", True)
    oh.remove_node(model, "/model.22/Concat", True)

# Save constants
const_name_subadd = "/model.22/Constant_9_output_0"
const_name_mul = "/model.22/Constant_12_output_0"
const_name_div = "/model.22/Constant_11_output_0"

addsub_const = oh.find_initializer_value(model, const_name_subadd)
mul_const = oh.find_initializer_value(model, const_name_mul)
print(f"Constant for Add and Sub: {addsub_const.shape}")
print(f"Constant for Mul: {mul_const.shape}")

# Modify bbox path.
cur_off = 0
for conv_idx in range(3):
    base_name = f"/model.22/cv2.{conv_idx}/cv2.{conv_idx}.2"
    old_conv_name = f"{base_name}/Conv"
    old_conv_node = oh.find_node(model, old_conv_name)

    old_conv_weight = oh.find_initializer_value(model, old_conv_node.input[1])
    old_conv_bias = oh.find_initializer_value(model, old_conv_node.input[2])

    mul_name = f"/model.22/cv2.{conv_idx}/cv2.{conv_idx}.1/act/Mul"
    mul_node = oh.find_node(model, mul_name)

    dfl_conv_nodes = [None]*4
    for split_idx in range(3, -1, -1):
        new_conv_name = f"{base_name}/{split_idx}/Conv"
        new_conv_weight_name = f"{new_conv_name}.weight"
        new_conv_bias_name = f"{new_conv_name}.bias"

        oh.add_initializer(model, new_conv_weight_name, old_conv_weight[16*split_idx:16*(split_idx+1), ...])
        oh.add_initializer(model, new_conv_bias_name, old_conv_bias[16*split_idx:16*(split_idx+1)])

        oh.insert_node(
            model,
            mul_node,
            new_conv_node := oh.make_node(
                name=new_conv_name,
                op_type="Conv",
                inputs=[mul_node.output[0], new_conv_weight_name, new_conv_bias_name],
                outputs=[f"{new_conv_name}_output"]
            ),
            insert_only=True
        )

        new_base_name = f"/model.22/dfl/{conv_idx}/{split_idx}"
        new_softmax_name = f"{new_base_name}/Softmax"
        oh.insert_node(
            model,
            new_conv_node,
            new_softmax_node := oh.make_node(
                name=new_softmax_name,
                op_type="Softmax",
                inputs=new_conv_node.output,
                outputs=[f"{new_softmax_name}_output"],
                axis=1
            ),
            insert_only=True
        )

        new_conv_name = f"{new_base_name}/Conv"
        oh.insert_node(
            model,
            new_softmax_node,
            new_conv_node := oh.make_node(
                name=new_conv_name,
                op_type="Conv",
                inputs=[new_softmax_node.output[0], "model.22.dfl.conv.weight"],
                outputs=[f"{new_conv_name}_output"]
            ),
            insert_only=True
        )
        dfl_conv_nodes[split_idx] = new_conv_node

    cur_h = H//(2**(conv_idx+3))
    cur_w = W//(2**(conv_idx+3))
    if bbox_opt_version == 1:
        new_base_name = f"/model.22/dfl/{conv_idx}"
        new_concat_name = f"{new_base_name}/Concat_0"
        oh.insert_node(
            model,
            dfl_conv_nodes[3],
            concat_0_node := oh.make_node(
                name=new_concat_name,
                op_type="Concat",
                inputs=[dfl_conv_nodes[0].output[0], dfl_conv_nodes[1].output[0]],
                outputs=[f"{new_concat_name}_output"],
                axis=1
            ),
            insert_only=True
        )
        new_concat_name = f"{new_base_name}/Concat_1"
        oh.insert_node(
            model,
            concat_0_node,
            concat_1_node := oh.make_node(
                name=new_concat_name,
                op_type="Concat",
                inputs=[dfl_conv_nodes[2].output[0], dfl_conv_nodes[3].output[0]],
                outputs=[f"{new_concat_name}_output"],
                axis=1
            ),
            insert_only=True
        )

        cur_addsub_const = addsub_const[..., cur_off:cur_off+cur_h*cur_w].reshape(1, 2, cur_h, cur_w)
        cur_mul_const = mul_const[..., cur_off:cur_off+cur_h*cur_w].reshape(1, cur_h, cur_w)
        cur_off += cur_h*cur_w

        new_sub_name = f"{new_base_name}/Sub_0"
        oh.add_initializer(model, f"{new_sub_name}/Const", cur_addsub_const)

        oh.insert_node(
            model,
            concat_1_node,
            sub_0_node := oh.make_node(
                name=new_sub_name,
                op_type="Sub",
                inputs=[f"{new_sub_name}/Const", concat_0_node.output[0]],
                outputs=[f"{new_sub_name}_output"]
            ),
            insert_only=True
        )

        new_add_name = f"{new_base_name}/Add_0"
        oh.add_initializer(model, f"{new_add_name}/Const", cur_addsub_const)

        oh.insert_node(
            model,
            sub_0_node,
            add_0_node := oh.make_node(
                name=new_add_name,
                op_type="Add",
                inputs=[f"{new_add_name}/Const", concat_1_node.output[0]],
                outputs=[f"{new_add_name}_output"]
            ),
            insert_only=True
        )

        new_add_name = f"{new_base_name}/Add_1"
        oh.insert_node(
            model,
            add_0_node,
            add_1_node := oh.make_node(
                name=new_add_name,
                op_type="Add",
                inputs=[sub_0_node.output[0], add_0_node.output[0]],
                outputs=[f"{new_add_name}_output"]
            ),
            insert_only=True
        )

        new_div_name = f"{new_base_name}/Div"
        oh.insert_node(
            model,
            add_1_node,
            div_node := oh.make_node(
                name=new_div_name,
                op_type="Div",
                inputs=[add_1_node.output[0], const_name_div],
                outputs=[f"{new_div_name}_output"]
            ),
            insert_only=True
        )

        new_sub_name = f"{new_base_name}/Sub_1"
        oh.insert_node(
            model,
            div_node,
            sub_1_node := oh.make_node(
                name=new_sub_name,
                op_type="Sub",
                inputs=[add_0_node.output[0], sub_0_node.output[0]],
                outputs=[f"{new_sub_name}_output"]
            ),
            insert_only=True
        )

        new_concat_name = f"{new_base_name}/Concat_2"
        oh.insert_node(
            model,
            sub_1_node,
            concat_node := oh.make_node(
                name=new_concat_name,
                op_type="Concat",
                inputs=[div_node.output[0], sub_1_node.output[0]],
                outputs=[f"{new_concat_name}_output"],
                axis=1
            ),
            insert_only=True
        )

        new_mul_name = f"{new_base_name}/Mul"
        oh.add_initializer(model, f"{new_mul_name}/Const", cur_mul_const)
        oh.insert_node(
            model,
            concat_node,
            oh.make_node(
                name=new_mul_name,
                op_type="Mul",
                inputs=[concat_node.output[0], f"{new_mul_name}/Const"],
                outputs=[f"bbox_{conv_idx}"]
            ),
            insert_only=True
        )
    else:
        new_base_name = f"/model.22/dfl/{conv_idx}"
        new_concat_name = f"{new_base_name}/Concat"
        oh.insert_node(
            model,
            dfl_conv_nodes[3],
            concat_node := oh.make_node(
                name=new_concat_name,
                op_type="Concat",
                inputs=[x.output[0] for x in dfl_conv_nodes],
                outputs=[f"{new_concat_name}_output"],
                axis=1
            ),
            insert_only=True
        )

        cur_mul_const = 2**(conv_idx+3)
        new_conv_name = f"{new_base_name}/Conv"
        conv_weight = np.array(
            [
                [-0.5, 0, 0.5, 0],
                [0, -0.5, 0, 0.5],
                [1, 0, 1, 0],
                [0, 1, 0, 1]
            ],
        ).reshape(4, 4, 1, 1) * cur_mul_const
        oh.add_initializer(model, f"{new_conv_name}.weight", conv_weight)

        oh.insert_node(
            model,
            concat_node,
            conv_node := oh.make_node(
                name=new_conv_name, 
                op_type="Conv",
                inputs=[concat_node.output[0], f"{new_conv_name}.weight"],
                outputs=[f"{new_conv_name}_output"]
            ),
            insert_only=True
        )

        new_add_name = f"{new_base_name}/Add"
        add_const = list()
        for i in range(4):
            add_const.append(list())
            for j in range(cur_h):
                add_const[i].append(list())
                for k in range(cur_w):
                    if i == 0:
                        add_const[i][j].append(0.5 + k)
                    elif i == 1:
                        add_const[i][j].append(0.5 + j)
                    else:
                        add_const[i][j].append(0)
        add_const = np.array(add_const).reshape(1, 4, cur_h, cur_w) * cur_mul_const
        oh.add_initializer(model, f"{new_add_name}/Const", add_const)
        oh.insert_node(
            model,
            conv_node,
            oh.make_node(
                name=new_add_name,
                op_type="Add",
                inputs=[conv_node.output[0], f"{new_add_name}/Const"],
                outputs=[f"bbox_{conv_idx}"]
            ),
            insert_only=True
        )

    oh.remove_node(model, old_conv_name, True)


# Modify class probability path.
for conv_idx in range(3):
    base_name = f"/model.22/cv3.{conv_idx}/cv3.{conv_idx}.2"
    conv_name = f"{base_name}/Conv"
    sigmoid_name = f"{base_name}/Sigmoid"
    conv_node = oh.find_node(model, conv_name)
    oh.insert_node(
        model,
        conv_node,
        oh.make_node(
            name=sigmoid_name,
            op_type="Sigmoid",
            inputs=conv_node.output,
            outputs=[f"class_prob_{conv_idx}"]
        ),
        insert_only=True
    )


# Remove all unneeded nodes.
oh.remove_node(model, "/model.22/Concat", True)
oh.remove_node(model, "/model.22/Reshape", True)

oh.remove_node(model, "/model.22/Concat_1", True)
oh.remove_node(model, "/model.22/Reshape_1", True)

oh.remove_node(model, "/model.22/Concat_2", True)
oh.remove_node(model, "/model.22/Reshape_2", True)


oh.remove_node(model, "/model.22/Concat_3", True)
oh.remove_node(model, "/model.22/Split", True)
oh.remove_node(model, "/model.22/Sigmoid", True)

oh.remove_node(model, "/model.22/dfl/Reshape", True)
oh.remove_node(model, "/model.22/dfl/Transpose", True)
oh.remove_node(model, "/model.22/dfl/Softmax", True)
oh.remove_node(model, "/model.22/dfl/conv/Conv", True)
oh.remove_node(model, "/model.22/dfl/Reshape_1", True)

oh.remove_node(model, "/model.22/Slice", True)
oh.remove_node(model, "/model.22/Slice_1", True)


oh.remove_node(model, "/model.22/Sub", True)
oh.remove_node(model, "/model.22/Add_1", True)
oh.remove_node(model, "/model.22/Add_2", True)
oh.remove_node(model, "/model.22/Sub_1", True)
oh.remove_node(model, "/model.22/Div_1", True)
oh.remove_node(model, "/model.22/Concat_4", True)
oh.remove_node(model, "/model.22/Mul_2", True)
oh.remove_node(model, "/model.22/Concat_5", True)


# Simplify and save model.
oh.save_model(model, f"{mod_model_name}.onnx")

