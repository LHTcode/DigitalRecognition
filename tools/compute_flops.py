from ptflops import get_model_complexity_info
from model.MobileNetv2 import MobileNetv2
from model.LeNet5 import LeNet5
from model.Linear import Linear
from config.global_config import global_config

def compute_flops(model):
    macs, params = get_model_complexity_info(
        model, (1, *global_config.INPUT_SIZE), as_strings=True,
        print_per_layer_stat=True, verbose=True
    )
    print("Computational complexity: {macs:s}".format(macs=macs))
    print("Number of parameters: {params:s}".format(params=params))
    return

if __name__ == '__main__':
    model = MobileNetv2(wid_mul=0.4, output_channels=global_config.CLASSES_NUM).to(global_config.DEVICE)
    # model = LeNet5(8, True).to(global_config.DEVICE)
    # model = Linear().to(global_config.DEVICE)
    compute_flops(model)