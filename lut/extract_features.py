import logging
import numpy as np
from sklearn.metrics import mean_squared_error


def get_accuracy(y_pred, y_true, threshold = 0.01):
    a = (y_true - y_pred) / y_true
    b = np.where(abs(a) <= threshold)
    return len(b[0]) / len(y_true)


def latency_metrics(y_pred, y_true):
    """
    evaluation metrics for prediction performance
    """
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    rmspe = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
    rmse = np.sqrt(mean_squared_error(y_pred, y_true))
    acc5 = get_accuracy(y_pred, y_true, threshold=0.05)
    acc10 = get_accuracy(y_pred, y_true, threshold=0.10)
    acc15 = get_accuracy(y_pred, y_true, threshold=0.15)
    return rmse, rmspe, rmse / np.mean(y_true), acc5, acc10, acc15


def get_conv_flop_params(hw, cin, cout, kernel_size, stride):
    params = cout * (kernel_size * kernel_size * cin + 1)
    flops = 2 * hw / stride * hw / stride * params
    return flops, params


def get_dwconv_flop_params(hw, cout, kernel_size, stride):
    params = cout * (kernel_size * kernel_size + 1)
    flops = 2 * hw / stride * hw / stride * params
    return flops, params


def get_fc_flop_params(cin, cout):
    params = (2 * cin + 1) * cout
    flops = params
    return flops, params


def get_flops_params(kernel_type, hw, cin, cout, kernelsize, stride):
    if "dwconv" in kernel_type:
        return get_dwconv_flop_params(hw, cout, kernelsize, stride)
    elif "conv" in kernel_type:
        return get_conv_flop_params(hw, cin, cout, kernelsize, stride)
    elif "fc" in kernel_type:
        return get_fc_flop_params(cin, cout)


def get_predict_features(config):
    """
    get prediction features
    """
    mdicts = {}
    layer = 0
    for item in config:
        logging.info(item)
    for item in config:
        op = item["op"]
        if "conv" in op or "maxpool" in op or "avgpool" in op:
            cout = item["cout"]
            cin = item["cin"]
            ks = item["ks"][1]
            s = item["strides"][1] if "strides" in item else 1
            inputh = item["inputh"]
        if op in ["channelshuffle", "split"]:
            [b, inputh, inputw, cin] = item["input_tensors"][0]

        if "conv" in op:
            flops, params = get_flops_params(op, inputh, cin, cout, ks, s)
            features = [inputh, cin, cout, ks, s, flops / 2e6, params / 1e6]
        elif "fc" in op or "fc-relu" in op:
            cout = item["cout"]
            cin = item["cin"]
            flop = (2 * cin + 1) * cout
            features = [cin, cout, flop / 2e6, flop / 1e6]
        elif "pool" in op and "global" not in op:
            features = [inputh, cin, cout, ks, s]
        elif "global-pool" in op or "global-avgpool" in op or "gap" in op:
            inputh = item["inputh"] if hasattr(item, "inputh") else 1
            cin = item["cin"]
            features = [inputh, cin]
        elif "channelshuffle" in op:
            features = [inputh, cin]
        elif "split" in op:
            features = [inputh, cin]
        elif "se" in op or "SE" in op:
            inputh = item["input_tensors"][-1][-2]
            cin = item["input_tensors"][-1][-1]
            features = [inputh, cin]
        elif "concat" in op:  # maximum 4 branches
            itensors = item["input_tensors"]
            inputh = itensors[0][1]
            features = [inputh, len(itensors)]
            for it in itensors:
                co = it[-1]
                features.append(co)
            if len(features) < 6:
                features = features + [0] * (6 - len(features))
            elif len(features) > 6:
                nf = features[0:6]
                features = nf
                features[1] = 6
        elif op in ["hswish"]:
            if "inputh" in item:
                inputh = item["inputh"]
            else:
                if len(item["input_tensors"][0]) == 2:
                    inputh = item["input_tensors"][0][0]
                else:
                    inputh = item["input_tensors"][0][1]
            cin = item["cin"]
            features = [inputh, cin]
        elif op in ["bn", "relu", "bn-relu"]:
            itensors = item["input_tensors"]
            if len(itensors[0]) == 4:
                inputh = itensors[0][1]
                cin = itensors[0][3]
            else:
                inputh = itensors[0][0]
                cin = itensors[0][1]
            features = [inputh, cin]

        elif op in ["add-relu", "add"]:
            itensors = item["input_tensors"]
            inputh = itensors[0][1]
            cin1 = itensors[0][3]
            cin2 = itensors[1][3]
            features = [inputh, cin1, cin2]
        else: # indicates that there is no matching predictor for this op
            # logging.warning(f'There is no matching predictor for op {op}.')
            continue
        mdicts[layer] = {}
        mdicts[layer][op] = features
        layer += 1
    return mdicts


def merge_conv_kernels(kernelname):
    """
    to speed up, we merge conv and dwconv related kernels into one kernel by their name
    """
    if "conv" in kernelname and "dwconv" not in kernelname:
        return "conv-bn-relu"
    elif "dwconv" in kernelname:
        return "dwconv-bn-relu"
    else:
        return kernelname


def get_kernel_name(optype):
    """
    for many similar kernels, we use one kernel predictor since their latency difference is negligible,
    return the kernel name via the optype
    """
    if "conv" in optype and "dwconv" not in optype:
        optype = "conv-bn-relu"
    if "dwconv" in optype:
        optype = "dwconv-bn-relu"
    if optype == "fc-relu":
        optype = "fc"
    if optype == "max-pool":
        optype = "maxpool"
    if optype == "avg-pool":
        optype = "avgpool"
    if optype in ["global-pool", "gap"]:
        optype = "global-avgpool"
    if optype == "channel_shuffle":
        optype = "channelshuffle"
    if optype in ["bn-relu"]:
        optype = "bnrelu"
    if optype in ["add-relu"]:
        optype = "addrelu"

    if optype in ["SE", "SE-relu", "se", "se-relu"]:
        optype = "se"

    return optype


def predict_model(model, predictors):
    """
    @params:
    model: the model config with prediction features
    predictors: loaded pkl predictors
    """
    py = 0
    dicts = {}
    for layer in model:
        kernel = list(model[layer].keys())[0]
        features = model[layer][kernel]
        rkernel = merge_conv_kernels(kernel)
        if rkernel not in dicts:
            dicts[rkernel] = []
        dicts[rkernel].append(features)

    for kernel in dicts:
        kernelname = get_kernel_name(kernel)
        if kernelname in predictors:
            pred = predictors[kernelname]
            pys = pred.predict(dicts[kernel]) # in unit of ms
            if len(pys) != 0:
                py += sum(pys)

    return py
