"""
Credit: https://github.com/Yu-Zhewen/fpgaconvnet-torch

"""


import os


def initialize_wrapper(model_name, dataset_path, batch_size, workers):
    os.environ['DATA_PATH'] = dataset_path

    from models.yolo import UltralyticsModelWrapper
    model_wrapper = UltralyticsModelWrapper(model_name)


    if model_wrapper is None:
        raise NotImplementedError("Unknown dataset/model combination")

    model_wrapper.load_data(batch_size, workers)
    model_wrapper.load_model()

    return model_wrapper
