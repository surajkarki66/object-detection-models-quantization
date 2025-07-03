import os
import pathlib
import random

import torch

from models import initialize_wrapper
from quantization import QuantMode, quantize_model


def main(model_name, dataset_path, gpu=False):

    output_path = os.path.join(os.getcwd(), f"output/model_name")
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    if gpu is not None:
        torch.cuda.set_device("cuda")

    random.seed(0)
    torch.manual_seed(0)

    model_wrapper = initialize_wrapper(model_name, os.path.expanduser(dataset_path), 32, 8)

    # Inference
    print("FLOAT32 Inference")
    model_wrapper.inference("test")
    model_wrapper.generate_onnx_files(
        os.path.join(output_path, "float32"))

    # FP16
    print("NETWORK FP16 Inference")
    # reload the model everytime a new quantization mode is tested
    model_wrapper.load_model()
    quantize_model(model_wrapper, {
                   'weight_width': 16, 'data_width': 16, 'mode': QuantMode.NETWORK_FP})
    model_wrapper.inference("test")
    model_wrapper.generate_onnx_files(
        os.path.join(output_path, "network_fp16"))

    # FP8
    print("NETWORK FP8 Inference")
    model_wrapper.load_model()
    quantize_model(model_wrapper, {
                   'weight_width': 8, 'data_width': 8, 'mode': QuantMode.NETWORK_FP})
    model_wrapper.inference("test")
    model_wrapper.generate_onnx_files(
        os.path.join(output_path, "network_fp8"))

    # BFP8
    print("LAYER BFP8 Inference")
    model_wrapper.load_model()
    quantize_model(model_wrapper, {
                   'weight_width': 8, 'data_width': 8, 'mode': QuantMode.LAYER_BFP})
    model_wrapper.inference("test")
    model_wrapper.generate_onnx_files(
        os.path.join(output_path, "layer_bfp8"))

    # BFP8 - Channel wise
    print("CHANNEL BFP8 Inference")
    # note: CHANNEL_BFP can be worse than LAYER_BFP, if calibration size is small!
    model_wrapper.load_model()
    quantize_model(model_wrapper,  {
                   'weight_width': 8, 'data_width': 8, 'mode': QuantMode.CHANNEL_BFP})
    model_wrapper.inference("test")
    model_wrapper.generate_onnx_files(
        os.path.join(output_path, "channel_bfp8"))


if __name__ == '__main__':
    model_name = "./models/yolov8n.pt"
    dataset_path = "../../Data/"
    gpu=True
    main(model_name, dataset_path, gpu)
