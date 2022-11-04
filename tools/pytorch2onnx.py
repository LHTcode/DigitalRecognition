import torch.onnx
import torch

from model.MobileNetv2 import MobileNetv2
from data.classes import NUMBER_CLASSES, CLASSES_NAME
from pathlib import Path

def convert_ONNX(model, param_dir, input_size):
    param_file = Path(param_dir) / 'best_model.pt'
    model_params = torch.load(param_file, map_location="cuda:0")
    model.load_state_dict(model_params['state'])
    model.eval()
    dummy_input = torch.randn(1, *input_size, requires_grad=True)
    torch.onnx.export(model,  # model_params being run
                      dummy_input,  # model_params input (or a tuple for multiple inputs)
                      str(Path(param_dir) / "digitalClassifier.onnx"),  # where to save the model_params
                      export_params=True,  # store the trained parameter weights inside the model_params file
                      opset_version=11,  # the ONNX version to export the model_params to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['digitalInput'],  # the model_params's input names
                      output_names=['digitalOutput'],  # the model_params's output names
                      dynamic_axes={'digitalInput': {0: 'batch_size'},  # variable length axes
                                    'digitalOutput': {0: 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')

if __name__ == '__main__':
    root_path = Path.cwd().parent
    model = MobileNetv2(wid_mul=0.5, output_channels=NUMBER_CLASSES)
    param_dir = root_path / 'model_params'
    convert_ONNX(model, str(param_dir), (1, 28, 28))