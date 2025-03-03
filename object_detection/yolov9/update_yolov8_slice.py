
import numpy as np
from update_onnx_utils import *

# model_name = "Yolov8_slice"
model_name = "yolov8n-seg"
mod_model_name = f"{model_name}_mod"

H, W = 480, 640

model = load_model(model_name)

# Remove all outputs and reconstruct outputs.
remove_output(model)
add_output(model, "bbox_0", (1, 4, H//8, W//8))
add_output(model, "bbox_1", (1, 4, H//16, W//16))
add_output(model, "bbox_2", (1, 4, H//32, W//32))
add_output(model, "class_prob_0", (1, 87, H//8, W//8))
add_output(model, "class_prob_1", (1, 87, H//16, W//16))
add_output(model, "class_prob_2", (1, 87, H//32, W//32))
add_output(model, "mask_coeff_0", (1, 32, H//8, W//8))
add_output(model, "mask_coeff_1", (1, 32, H//16, W//16))
add_output(model, "mask_coeff_2", (1, 32, H//32, W//32))
add_output(model, "mask", (1, 32, H//4, W//4))

# Modify mask path.
connect_output(model, "/module_22/proto/cv3/act/Mul", "mask")

# Modify mask coeff path.
connect_output(model, "/module_22/cv4.0/cv4.0.2/Conv", "mask_coeff_0")
connect_output(model, "/module_22/cv4.1/cv4.1.2/Conv", "mask_coeff_1")
connect_output(model, "/module_22/cv4.2/cv4.2.2/Conv", "mask_coeff_2")
remove_node(model, "/module_22/Reshape")
remove_node(model, "/module_22/Reshape_1")
remove_node(model, "/module_22/Reshape_2")
remove_node(model, "/module_22/Concat")


# Modify bbox path.
bbox_version = 2
addsub_const = find_initializer_value(model, "/module_22/Constant_22_output_0")
mul_const = find_initializer_value(model, "/module_22/Constant_25_output_0")
cur_off = 0
for conv_idx in range(3):
    base_name = f"/module_22/cv2.{conv_idx}/cv2.{conv_idx}.2"
    old_conv_name = f"{base_name}/Conv"
    old_conv_node = find_node(model, old_conv_name)

    old_conv_weight = find_initializer_value(model, old_conv_node.input[1])
    old_conv_bias = find_initializer_value(model, old_conv_node.input[2])

    mul_name = f"/module_22/cv2.{conv_idx}/cv2.{conv_idx}.1/act/Mul"
    mul_node = find_node(model, mul_name)

    dfl_conv_nodes = [None]*4
    for split_idx in range(3, -1, -1):
        new_conv_name = f"{base_name}/{split_idx}/Conv"
        new_conv_weight_name = f"{new_conv_name}.weight"
        new_conv_bias_name = f"{new_conv_name}.bias"

        add_initializer(model, new_conv_weight_name, old_conv_weight[16*split_idx:16*(split_idx+1), ...])
        add_initializer(model, new_conv_bias_name, old_conv_bias[16*split_idx:16*(split_idx+1)])

        insert_node(
            model,
            mul_node,
            new_conv_node := make_node(
                name=new_conv_name,
                op_type="Conv",
                inputs=[mul_node.output[0], new_conv_weight_name, new_conv_bias_name],
                outputs=[f"{new_conv_name}_output"]
            ),
            insert_only=True
        )

        new_base_name = f"/module_22/dfl/{conv_idx}/{split_idx}"
        new_softmax_name = f"{new_base_name}/Softmax"
        insert_node(
            model,
            new_conv_node,
            new_softmax_node := make_node(
                name=new_softmax_name,
                op_type="Softmax",
                inputs=new_conv_node.output,
                outputs=[f"{new_softmax_name}_output"],
                axis=1
            ),
            insert_only=True
        )

        new_conv_name = f"{new_base_name}/Conv"
        insert_node(
            model,
            new_softmax_node,
            new_conv_node := make_node(
                name=new_conv_name,
                op_type="Conv",
                inputs=[new_softmax_node.output[0], "module_22.dfl.conv.weight"],
                outputs=[f"{new_conv_name}_output"]
            ),
            insert_only=True
        )
        dfl_conv_nodes[split_idx] = new_conv_node

    cur_h = H//(2**(conv_idx+3))
    cur_w = W//(2**(conv_idx+3))
    if bbox_version == 1:
        new_base_name = f"/module_22/dfl/{conv_idx}"
        new_concat_name = f"{new_base_name}/Concat_0"
        insert_node(
            model,
            dfl_conv_nodes[3],
            concat_0_node := make_node(
                name=new_concat_name,
                op_type="Concat",
                inputs=[dfl_conv_nodes[0].output[0], dfl_conv_nodes[1].output[0]],
                outputs=[f"{new_concat_name}_output"],
                axis=1
            ),
            insert_only=True
        )
        new_concat_name = f"{new_base_name}/Concat_1"
        insert_node(
            model,
            concat_0_node,
            concat_1_node := make_node(
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
        add_initializer(model, f"{new_sub_name}/Const", cur_addsub_const)

        insert_node(
            model,
            concat_1_node,
            sub_0_node := make_node(
                name=new_sub_name,
                op_type="Sub",
                inputs=[f"{new_sub_name}/Const", concat_0_node.output[0]],
                outputs=[f"{new_sub_name}_output"]
            ),
            insert_only=True
        )

        new_add_name = f"{new_base_name}/Add_0"
        add_initializer(model, f"{new_add_name}/Const", cur_addsub_const)

        insert_node(
            model,
            sub_0_node,
            add_0_node := make_node(
                name=new_add_name,
                op_type="Add",
                inputs=[f"{new_add_name}/Const", concat_1_node.output[0]],
                outputs=[f"{new_add_name}_output"]
            ),
            insert_only=True
        )

        new_add_name = f"{new_base_name}/Add_1"
        insert_node(
            model,
            add_0_node,
            add_1_node := make_node(
                name=new_add_name,
                op_type="Add",
                inputs=[sub_0_node.output[0], add_0_node.output[0]],
                outputs=[f"{new_add_name}_output"]
            ),
            insert_only=True
        )

        new_div_name = f"{new_base_name}/Div"
        insert_node(
            model,
            add_1_node,
            div_node := make_node(
                name=new_div_name,
                op_type="Div",
                inputs=[add_1_node.output[0], "/module_22/Constant_24_output_0"],
                outputs=[f"{new_div_name}_output"]
            ),
            insert_only=True
        )

        new_sub_name = f"{new_base_name}/Sub_1"
        insert_node(
            model,
            div_node,
            sub_1_node := make_node(
                name=new_sub_name,
                op_type="Sub",
                inputs=[add_0_node.output[0], sub_0_node.output[0]],
                outputs=[f"{new_sub_name}_output"]
            ),
            insert_only=True
        )

        new_concat_name = f"{new_base_name}/Concat_2"
        insert_node(
            model,
            sub_1_node,
            concat_node := make_node(
                name=new_concat_name,
                op_type="Concat",
                inputs=[div_node.output[0], sub_1_node.output[0]],
                outputs=[f"{new_concat_name}_output"],
                axis=1
            ),
            insert_only=True
        )

        new_mul_name = f"{new_base_name}/Mul"
        add_initializer(model, f"{new_mul_name}/Const", cur_mul_const)
        insert_node(
            model,
            concat_node,
            make_node(
                name=new_mul_name,
                op_type="Mul",
                inputs=[concat_node.output[0], f"{new_mul_name}/Const"],
                outputs=[f"bbox_{conv_idx}"]
            ),
            insert_only=True
        )
    else:
        new_base_name = f"/module_22/dfl/{conv_idx}"
        new_concat_name = f"{new_base_name}/Concat"
        insert_node(
            model,
            dfl_conv_nodes[3],
            concat_node := make_node(
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
        add_initializer(model, f"{new_conv_name}.weight", conv_weight)

        insert_node(
            model,
            concat_node,
            conv_node := make_node(
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
        add_initializer(model, f"{new_add_name}/Const", add_const)
        insert_node(
            model,
            conv_node,
            make_node(
                name=new_add_name,
                op_type="Add",
                inputs=[conv_node.output[0], f"{new_add_name}/Const"],
                outputs=[f"bbox_{conv_idx}"]
            ),
            insert_only=True
        )

    remove_node(model, old_conv_name)


# Modify class probability path.
for conv_idx in range(3):
    base_name = f"/module_22/cv3.{conv_idx}/cv3.{conv_idx}.2"
    conv_name = f"{base_name}/Conv"
    sigmoid_name = f"{base_name}/Sigmoid"
    conv_node = find_node(model, conv_name)
    insert_node(
        model,
        conv_node,
        make_node(
            name=sigmoid_name,
            op_type="Sigmoid",
            inputs=conv_node.output,
            outputs=[f"class_prob_{conv_idx}"]
        ),
        insert_only=True
    )


# Remove all unneeded nodes.
remove_node(model, "/module_22/Slice_1")
remove_node(model, "/module_22/Sigmoid")
remove_node(model, "/module_22/Concat_1")
remove_node(model, "/module_22/Concat_2")
remove_node(model, "/module_22/Concat_3")
remove_node(model, "/module_22/Concat_4")
remove_node(model, "/module_22/Reshape_3")
remove_node(model, "/module_22/Reshape_4")
remove_node(model, "/module_22/Reshape_5")
remove_node(model, "/module_22/Slice")
remove_node(model, "/module_22/dfl/Reshape")
remove_node(model, "/module_22/dfl/Transpose")
remove_node(model, "/module_22/dfl/Softmax")
remove_node(model, "/module_22/dfl/conv/Conv")
remove_node(model, "/module_22/dfl/Reshape_1")
remove_node(model, "/module_22/Slice_2")
remove_node(model, "/module_22/Slice_3")
remove_node(model, "/module_22/Add")
remove_node(model, "/module_22/Add_1")
remove_node(model, "/module_22/Sub")
remove_node(model, "/module_22/Sub_1")
remove_node(model, "/module_22/Div")
remove_node(model, "/module_22/Concat_5")
remove_node(model, "/module_22/Mul")
remove_node(model, "/module_22/Concat_6")


# Simplify and save model.
save_model(model, mod_model_name)
