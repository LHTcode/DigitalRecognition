import torch.onnx
import torch
from pathlib import Path

from model.MobileNetv2 import MobileNetv2
from model.LeNet import LeNet
from data.classes import NUMBER_CLASSES, CLASSES_NAME
from config.global_config import global_config

def convert_ONNX(model, param_dir, input_size:tuple):
    param_file = Path(param_dir) / (global_config.MODEL_NAME + '.pt')
    model_params = torch.load(param_file, map_location="cuda:0")
    model.load_state_dict(model_params['state'])
    model.eval()
    dummy_input = torch.randn(1, (1, *input_size), requires_grad=True)
    torch.onnx.export(model,  # model_params being run
                      dummy_input,  # model_params input (or a tuple for multiple inputs)
                      str(Path(param_dir) / (global_config.MODEL_NAME + ".onnx")),  # where to save the model_params
                      export_params=True,  # store the trained parameter weights inside the model_params file
                      opset_version=11,  # the ONNX version to export the model_params to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['digitalInput'],  # the model_params's input names
                      output_names=['digitalOutput'],  # the model_params's output names
                      dynamic_axes={'digitalInput': {0: 'batch_size'},  # variable length axes
                                    'digitalOutput': {0: 'batch_size'}})
    print(" ")
    print(f'Model {global_config.MODEL_NAME} has been converted to ONNX')
    return

if __name__ == '__main__':
    root_path = Path.cwd().parent
    model = MobileNetv2(wid_mul=1, output_channels=NUMBER_CLASSES)
    # model = LeNet(global_config.CLASSES_NUM)
    param_dir = root_path / 'model_params'
    convert_ONNX(model, str(param_dir), global_config.INPUT_SIZE)