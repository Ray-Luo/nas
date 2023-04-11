import os
import yaml
import pickle
import logging
from glob import glob
import torch

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
        try:
            logging.info("NNI-based Torch Converter is applied for model conversion")
            converter = NNIBasedTorchConverter(model, args)
        except:
            raise NotImplementedError("Your model is not fully converted by NNI-based converter. Please set apply_nni=False and try again.")
    else:
        # apply Onnx-based torch converter, which requires onnx installation (well tested version is onnx==1.9.0)
        # and the conversion is more stable
        logging.info("Onnx-based Torch Converter is applied for model conversion")
        converter = OnnxBasedTorchConverter(model, args)
    return converter.convert()


def model_file_to_graph(filename: str, model_type: str, input_shape=(1, 3, 224, 224), apply_nni=False):
    models = try_import_torchvision_models()
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




class nnMeterPredictor:
    def __init__(self, predictors, fusionrule):
        self.kernel_predictors = predictors
        self.fusionrule = fusionrule
        # self.kd = KernelDetector(self.fusionrule)

    def predict(
        self, model, model_type, input_shape=(1, 3, 224, 224), apply_nni=False
    ):
        """
        return the predicted latency in microseconds (ms)
        @params:
        model: the model to be predicted, allowed file include
            - the path to a saved tensorflow model file (*.pb), `model_type` must be set to "pb"
            - pytorch model object (nn.Module), `model_type` must be set to "torch"
            - ONNX model object or the path to a saved ONNX model file (*.onnx), `model_type` must be set to "onnx"
            - dictionary object following nn-Meter-IR format, `model_type` must be set to "nnmeter-ir"
            - dictionary object following NNI-IR format, `model_type` must be set to "nni-ir"

        model_type: string to specify the type of parameter model, allowed items are ["pb", "torch", "onnx", "nnmeter-ir", "nni-ir"]

        input_shape: the shape of input tensor for inference (if necessary), a random tensor according to the shape will be generated and used. This parameter is only
        accessed when model_type == 'torch'
        apply_nni: switch the torch converter used for torch model parsing. If apply_nni==True, NNI-based converter is used for torch model conversion, which requires
            nni>=2.4 installation and should use nn interface from NNI `import nni.retiarii.nn.pytorch as nn` to define the PyTorch modules. Otherwise Onnx-based torch
            converter is used, which requires onnx installation (well tested version is onnx>=1.9.0). NNI-based converter is much faster while the conversion is unstable
            as it could fail in some case. Onnx-based converter is much slower but stable compared to NNI-based converter. This parameter is only accessed when
            model_type == 'torch'
        """
        logging.info("Start latency prediction ...")
        if isinstance(model, str):
            graph = model_file_to_graph(model, model_type, input_shape, apply_nni=apply_nni)
        else:
            graph = model_to_graph(model, model_type, input_shape=input_shape, apply_nni=apply_nni)

        # logging.info(graph)
        self.kd.load_graph(graph)

        py = nn_predict(self.kernel_predictors, self.kd.get_kernels()) # in unit of ms
        logging.info(f"Predict latency: {py} ms")
        return py
