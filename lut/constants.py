CONV_TYPE = "Conv"
BN_TYPE = "BatchNormalization"
SLICE_TYPE = "Slice"
CONCAT_TYPE = "Concat"
MAXPOOL_TYPE = "MaxPool"
AVGPOOL_TYPE = "AveragePool"
RELU_TYPE = "Relu"
ADD_TYPE = "Add"
FC_TYPE = "Gemm"
RESHAPE_TYPE = "Reshape"
GAP_TYPE = "GlobalAveragePool"
CLIP_TYPE = "Clip"
MUL_TYPE = "Mul"
DIV_TYPE = "Div"
HARDSIGMOID_TYPE = "HardSigmoid"
FLATTEN_TYPE = "Flatten"
TRANSPOSE_TYPE = "Transpose"
REDUCEMEAN_TYPE = "ReduceMean"
SPLIT_TYPE = "Split"
PAD_TYPE = "Pad"


DUMMY_TYPES = [
    "Const",
    "Identity",
    "Placeholder",
]

# TODO: Refactor opset map. Should be moved to corresponding module.
OP_ALIAS = {
    # Tensorflow
    "Relu6": "relu",
    "Relu": "relu",
    "Add": "add",
    "Biasadd": "add",
    "Conv2D": "conv",
    "Reshape": "reshape",
    "FusedBatchNorm": "bn",
    "FusedBatchNormV3": "bn",
    "MatMul": "fc",
    "MaxPool": "maxpool",
    "AvgPool": "avgpool",
    "Mean": "gap",
    "Mul": "mul",
    "DepthwiseConv2dNative": "dwconv",
    "ConcatV2": "concat",
    "Split": "split",
    # ONNX
    "Conv": "conv",
    "BatchNormalization": "bn",
    "Slice": "split",
    "Concat": "concat",
    "AveragePool": "avgpool",
    "Relu": "relu",
    "Add": "add",
    "Gemm": "fc",
    "GlobalAveragePool": "gap",
    "Clip": "relu",
    "Mul": "mul",
    "Div": "div",
    "HardSigmoid": "hardsigmoid",
    "Flatten": "reshape",
    "Transpose": "transpose",
    "ReduceMean": "gap",
    "Split": "split",
    "Pad": "pad",
}
