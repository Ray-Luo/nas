import os
import yaml
import pickle
import logging
from glob import glob
import torch
import onnx
from itertools import chain
from .constants import SLICE_TYPE
from .extract_features import get_predict_features, predict_model
import tempfile
from onnxsim import simplify


def get_tensor_shape(tensor):
    shape = []
    for dim in tensor.type.tensor_type.shape.dim:
        shape.append(dim.dim_value)
    if len(shape) == 4:
        shape = [shape[0], shape[2], shape[3], shape[1]]
    return shape


class OnnxConverter:
    def __init__(self, model):
        from onnx import shape_inference
        inferred_model = shape_inference.infer_shapes(model)
        self.graph = inferred_model.graph

        self.tensors = {}
        for tensor in chain(self.graph.input, self.graph.value_info, self.graph.output):
            self.tensors[tensor.name] = {
                "shape": get_tensor_shape(tensor),
                "inputs": [],
                "outputs": [],
            }

        for node in self.graph.node:
            for input_name in node.input:
                if input_name in self.tensors:
                    self.tensors[input_name]["outputs"].append(node)
            for output_name in node.output:
                if output_name in self.tensors:
                    self.tensors[output_name]["inputs"].append(node)

    def fetch_attrs(self, node):
        from onnx import AttributeProto
        attrs = {}
        input_tensors = []
        for input_name in node.input:
            if input_name in self.tensors:
                input_tensors.append(self.tensors[input_name]["shape"])
        output_tensors = []
        for output_name in node.output:
            if output_name in self.tensors:
                output_tensors.append(self.tensors[output_name]["shape"])
        if node.op_type == SLICE_TYPE:
            for tensor_name in self._get_sibling_slice_output_tensors(node):
                output_tensors.append(self.tensors[tensor_name]["shape"])
        if (
            len(input_tensors) == 0
            or len(input_tensors[0]) <= 1
            or len(output_tensors) == 0
            or len(output_tensors[0]) <= 1
        ):
            logging.warning(f"Empty shape information with {node.name}")
            return attrs

        attrs["attr"] = {}
        attrs["type"] = node.op_type
        attrs["input_shape"] = input_tensors
        attrs["output_shape"] = output_tensors
        for attr in node.attribute:
            if attr.type == AttributeProto.FLOAT:
                attrs["attr"][attr.name] = attr.f
            elif attr.type == AttributeProto.INT:
                attrs["attr"][attr.name] = attr.i
            elif attr.type == AttributeProto.INTS:
                attrs["attr"][attr.name] = list(attr.ints)
            elif attr.type == AttributeProto.STRING:
                attrs["attr"][attr.name] = str(attr.s)
            else:
                logging.warning(f"Unsupported attributes type: {attr.type}")

        return attrs

    def convert(self):
        result = {}

        sliced_tensors = set()
        selected_slice = set()
        for node in self.graph.node:
            if node.op_type == SLICE_TYPE:
                tensor = node.input[0]
                if tensor in sliced_tensors:
                    continue
                else:
                    sliced_tensors.add(tensor)
                    selected_slice.add(node.name)

        for node in self.graph.node:
            outbounds = []
            inbounds = []
            if node.op_type == SLICE_TYPE and node.name not in selected_slice:
                continue

            for input_name in node.input:
                if input_name in self.tensors:  # remove dummy ops
                    for pred_pred in self.tensors[input_name]['inputs']:
                        inbounds.append(pred_pred.name)
            for output_name in node.output:
                if output_name in self.tensors:
                    for succ_succ in self.tensors[output_name]['outputs']:
                        outbounds.append(succ_succ.name)
                if node.op_type == SLICE_TYPE:
                    for tensor_name in self._get_sibling_slice_output_tensors(node):
                        outbounds.append(tensor_name)
                result[node.name] = {
                    "attr": self.fetch_attrs(node),
                    "outbounds": outbounds,
                    "inbounds": inbounds,
                }

        return result

    def _get_sibling_slice_output_tensors(self, node):
        output_tensors = []
        for slice in self.tensors[node.input[0]]["outputs"]:
            if slice.name != node.name and slice.op_type == SLICE_TYPE:
                for output_name in slice.output:
                    if output_name in self.tensors:
                        output_tensors.append(output_name)

        return output_tensors


class OnnxBasedTorchConverter(OnnxConverter):
    def __init__(self, model, example_inputs):
        with tempfile.TemporaryFile() as fp:
            torch.onnx.export(model, example_inputs, fp)
            fp.seek(0)
            model = onnx.load(fp, load_external_data=False)

        # convert model
        model_simp, check = simplify(model)

        assert check, "Simplified ONNX model could not be validated"
        super().__init__(model_simp)

def check_predictors(ppath, kernel_predictors):
    """
    @params:
    model: a pytorch/onnx/tensorflow model object or a str containing path to the model file
    """
    logging.info("checking local kernel predictors at " + ppath)
    if os.path.isdir(ppath):
        filenames = glob(os.path.join(ppath, "**.pkl"))
        # check if all the pkl files are included
        for kp in kernel_predictors:
            fullpath = os.path.join(ppath, kp + ".pkl")
            if fullpath not in filenames:
                return False
        return True
    else:
        return False

def loading_to_local(pred_info, dir):
    """ loading builtin predictors to local
    @params:
    pred_info: a dictionary containing predictor information
    dir: the local directory to store the kernel predictors and fusion rules
    """
    os.makedirs(dir, exist_ok=True)
    hardware = pred_info['name']
    ppath = os.path.join(dir, hardware)

    isdownloaded = check_predictors(ppath, pred_info["kernel_predictors"])
    if not isdownloaded:
        logging.keyinfo(f'Download from {pred_info["download"]} ...')
        # download_from_url(pred_info["download"], dir)

    # load predictors
    predictors = {}
    ps = glob(os.path.join(ppath, "**.pkl"))
    for p in ps:
        pname =  os.path.basename(p).replace(".pkl", "")
        with open(p, "rb") as f:
            logging.info("load predictor %s" % p)
            model = pickle.load(f)
            predictors[pname] = model
    fusionrule = os.path.join(ppath, "fusion_rules.json")
    # logging.info(fusionrule)
    if not os.path.isfile(fusionrule):
        raise ValueError(
            "check your fusion rule path, file " + fusionrule + " does not exist！"
        )
    return predictors, fusionrule


def loading_customized_predictor(pred_info={"name": "cortexA76cpu_tflite21","version":1.0}):
    """ loading customized predictor
    @params:
    pred_info: a dictionary containing predictor information
    """
    hardware = pred_info['name']
    ppath = pred_info['package_location']

    isexist = check_predictors(ppath, pred_info["kernel_predictors"])
    if not isexist:
        raise FileExistsError(f"The predictor {hardware} in {ppath} does not exist.")

    # load predictors
    predictors = {}
    ps = glob(os.path.join(ppath, "**.pkl"))
    for p in ps:
        pname =  os.path.basename(p).replace(".pkl", "")
        with open(p, "rb") as f:
            logging.info("load predictor %s" % p)
            model = pickle.load(f)
            predictors[pname] = model
    fusionrule = os.path.join(ppath, "fusion_rules.json")
    # logging.info(fusionrule)
    if not os.path.isfile(fusionrule):
        raise ValueError(
            "check your fusion rule path, file " + fusionrule + " does not exist！"
        )
    return predictors, fusionrule


def load_latency_predictor(predictor_name: str, predictor_version: float = None):
    """
    return the predictor model according to the given predictor name and version
    @params:
    predictor_name: string to specify the name of the target latency predictor. All built-in predictors can be viewed by nn_meter.list_latency_predictors()
        or through the config file in ~/.nn_meter/config/predictors.yaml.

    predictor_version: string to specify the version of the target latency predictor. If not specified (default as None), the lateast version of the
        predictor will be loaded.
    """
    user_data_folder = "/home/luoleuyouluole/.nn_meter/data/"
    # pred_info = load_predictor_config(predictor_name, predictor_version)
    pred_info = {"name": "cortexA76cpu_tflite21","version":1.0, "download":False}

    if "download" in pred_info:
        kernel_predictors, fusionrule = loading_to_local(pred_info, os.path.join(user_data_folder, 'predictor'))
    else:
        kernel_predictors, fusionrule = loading_customized_predictor(pred_info)

    return nnMeterPredictor(kernel_predictors, fusionrule)


def torch_model_to_graph(model, input_shape=(1, 3, 224, 224), apply_nni=False):
    args = torch.randn(*input_shape)
    try:
        # if the test model has no parameters (such as activation ops), there will be error when calling ``model.parameters``
        if next(model.parameters()).is_cuda:
            args = args.to("cuda")
    except:
        pass
    if apply_nni:
        # apply NNI-based torch converter, which requires nni>=2.4 installation and should use nn interface from NNI
        # `import nni.retiarii.nn.pytorch as nn` to define the PyTorch modules.
        pass
    else:
        # apply Onnx-based torch converter, which requires onnx installation (well tested version is onnx==1.9.0)
        # and the conversion is more stable
        logging.info("Onnx-based Torch Converter is applied for model conversion")
        converter = OnnxBasedTorchConverter(model, args)
    return converter.convert()


def model_file_to_graph(filename: str, model_type: str, input_shape=(1, 3, 224, 224), apply_nni=False):
    torchvision_zoo_dict = {
        'resnet18': 'models.resnet18()',
        'alexnet': 'models.alexnet()',
        'vgg16': 'models.vgg16()',
        'squeezenet': 'models.squeezenet1_0()',
        'densenet161': 'models.densenet161()',
        'inception_v3': 'models.inception_v3()',
        'googlenet': 'models.googlenet()',
        'shufflenet_v2': 'models.shufflenet_v2_x1_0()',
        'mobilenet_v2': 'models.mobilenet_v2()',
        'resnext50_32x4d': 'models.resnext50_32x4d()',
        'wide_resnet50_2': 'models.wide_resnet50_2()',
        'mnasnet': 'models.mnasnet1_0()',
    }
    if filename in torchvision_zoo_dict:
        model = eval(torchvision_zoo_dict[filename])
    else:
        suppost_list = ", ".join([k for k in torchvision_zoo_dict])
        raise ValueError(f"Unsupported model name: {filename} in torchvision. Supporting list: {suppost_list}")
    return torch_model_to_graph(model, input_shape, apply_nni)


def nn_predict(predictors, kernel_units):
    features = get_predict_features(kernel_units)
    py = predict_model(features, predictors)
    return py

class nnMeterPredictor:
    def __init__(self, predictors, fusionrule):
        self.kernel_predictors = predictors
        self.fusionrule = fusionrule
        # self.kd = KernelDetector(self.fusionrule)

    def predict(
        self, model, model_type, input_shape=(1, 3, 224, 224), apply_nni=False
    ):
        logging.info("Start latency prediction ...")
        if isinstance(model, str):
            graph = model_file_to_graph(model, model_type, input_shape, apply_nni=apply_nni)
        else:
            # graph = model_to_graph(model, model_type, input_shape=input_shape, apply_nni=apply_nni)
            pass

        # logging.info(graph)
        self.kd.load_graph(graph)

        py = nn_predict(self.kernel_predictors, self.kd.get_kernels()) # in unit of ms
        logging.info(f"Predict latency: {py} ms")
        return py
