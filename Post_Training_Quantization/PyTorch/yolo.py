"""
Credit: https://github.com/Yu-Zhewen/fpgaconvnet-torch

"""

import os

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn

from base import TorchModelWrapper

# note: do NOT move ultralytic import to the top, otherwise the edit in settings will not take effect

class UltralyticsModelWrapper(TorchModelWrapper):
    # https://github.com/ultralytics/ultralytics

    def load_model(self, eval=True):
        from ultralytics import YOLO
        self.yolo = YOLO(self.model_name)
        self.model = self.yolo.model
        self.model_fixer()
        self.input_size = (1, 3, self.yolo.overrides['imgsz'], self.yolo.overrides['imgsz'])

        # utlralytics conv bn fusion not working after compression, disable it
        def _fuse(verbose=True):
            return self.model
        self.model.fuse = _fuse

    def model_fixer(self):
        from ultralytics.nn.modules import Conv
        for name, module in self.named_modules():
            if isinstance(module, Conv) and isinstance(module.act, nn.SiLU):
                module.act = nn.Hardswish(inplace=True)

    def load_data(self, batch_size, workers):
        from ultralytics import settings

        DATA_PATH = os.environ.get("DATA_PATH")
        # set dataset path
        settings.update({'datasets_dir': DATA_PATH})

        # note: ultralytics automatically handle the dataloaders, only need to set the path
        self.data_loaders['calibrate'] = os.path.join(DATA_PATH, "data.yaml")
        self.data_loaders['validate'] = os.path.join(DATA_PATH, "data.yaml")

        self.batch_size = batch_size
        self.workers = workers

    def inference(self, mode="validate"):
        mode = "validate" if mode == "test" else mode
        print("Inference mode: {}".format(mode))
        self.yolo.model = self.model
        return self.yolo.val(batch=self.batch_size, workers=self.workers,
            device="cpu" if not torch.cuda.is_available() else f"cuda:{torch.cuda.current_device()}",
            data=self.data_loaders[mode], plots=False)

    def onnx_exporter(self, onnx_path):
        path = self.yolo.export(format="onnx", simplify=True, opset=14)
        os.rename(path, onnx_path)
        
        self.remove_detection_head_v8(onnx_path)

        # rename sideband info
        for method, info in self.sideband_info.items():
            new_info = {}
            for k, v in info.items():
                if k.startswith("yolo.model."):
                    new_info[k.replace("yolo.model.", "")] = v
                else:
                    new_info[k] = v
            self.sideband_info[method] = new_info

    def remove_detection_head_v8(self, onnx_path):
        graph = onnx.load(onnx_path)
        graph = gs.import_onnx(graph)

         # Names of reshape (post-processing) nodes to remove
        reshapes_to_remove = {
            "/model.22/Reshape",
            "/model.22/Reshape_1",
            "/model.22/Reshape_2"
        }

        # Remove reshape nodes by filtering
        graph.nodes = [node for node in graph.nodes if node.name not in reshapes_to_remove]

        # Get Concat nodes
        concat_names = [
            "/model.22/Concat",
            "/model.22/Concat_1",
            "/model.22/Concat_2"
        ]

        concat_nodes = {}
        for name in concat_names:
            node = next((n for n in graph.nodes if n.name == name), None)
            if node is None:
                raise ValueError(f"Concat node {name} not found in graph.")
            concat_nodes[name] = node

        # Update Resize nodes' ROI inputs
        for resize_name, roi_name in [("/model.10/Resize", "roi_0"), ("/model.13/Resize", "roi_1")]:
            resize_node = next((n for n in graph.nodes if n.name == resize_name), None)
            if resize_node is None:
                raise ValueError(f"Resize node {resize_name} not found.")
            if len(resize_node.inputs) > 1:
                resize_node.inputs[1] = gs.Constant(roi_name, np.array([0.0, 0.0, 0.0, 0.0]))

        # Create new graph outputs based on concat outputs
        graph.outputs = []
        for name, node in concat_nodes.items():
            output_name = f"{name}_output_0"
            output_var = gs.Variable(output_name, shape=node.outputs[0].shape, dtype=np.float32)
            node.outputs = [output_var]
            graph.outputs.append(output_var)

        # Clean up graph and export
        graph.cleanup()
        onnx_graph = gs.export_onnx(graph)
        onnx_graph.ir_version = 8  # Downgrade IR version for compatibility
        onnx.save(onnx_graph, onnx_path)
