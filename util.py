from math import ceil
import custom_op
import torch.nn as nn
import re
from custom_op.conv_avg import Conv2dAvg

def freeze_layers(module, freeze_cfgs):
    if not isinstance(freeze_cfgs, list):
        print("No Freeze Required")
        return
    for cfg in freeze_cfgs:
        path = cfg['path'].split(' ')
        layer = module
        for p in path:
            if p.startswith('[') and p.endswith(']'):
                if p[1:-1].isdigit():
                    layer = layer[int(p[1:-1])]
                else:
                    layer = layer[p[1:-1]]
            else:
                layer = getattr(layer, p)
        layer.eval()
        for param in layer.parameters():
            param.requires_grad = False


def grad_logger(dst, name):
    def hook(grad):
        dst[name] = grad
    return hook

## Hàm của mình

class Conv2dSizeHook:
    def __init__(self):
        self.input_size = {}
        self.output_size = {}

    def __call__(self, module, input, output):
        self.input_size[module] = input[0].shape #input[0].size()[2:]
        self.output_size[module] = output.shape

def get_all_conv(model):
    conv_layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.modules.conv.Conv2d) or isinstance(mod, Conv2dAvg): # Nếu mod là Conv2d hoặc Conv2dAvg
            conv_layers.append(mod)
    return conv_layers

def get_active_conv(model, freeze_cfgs):
    if freeze_cfgs == None:
        return get_all_conv(model)
    else:
        list_freeze_cfgs = []
        for cfg in freeze_cfgs:
            path = cfg['path'].replace(" ", ".")
            if '[' in path or ']' in path:
                path = path.replace("[", "").replace("]", "")
            list_freeze_cfgs.append(path)

        active_conv_layers = []
        # active_conv_layers = {}
        for name, mod in model.named_modules():
            if isinstance(mod, nn.modules.conv.Conv2d) or isinstance(mod, Conv2dAvg): # Nếu mod là Conv2d hoặc Conv2dAvg
                if not any(re.match(f"^{prefix}(?:\\.|$)", name) for prefix in list_freeze_cfgs): # Module đang xét không thuộc danh sách freeze
                    active_conv_layers.append(mod)
                    # active_conv_layers[name] = mod
        return active_conv_layers

def get_total_weight_size(model, element_size=4): # element_size = 4 bytes
    def _is_depthwise_conv(conv):
        return conv.groups == conv.in_channels == conv.out_channels
    
    conv_layers = get_all_conv(model)

    this_num_weight = 0
    for conv_layer in conv_layers:
        # if "Conv2dAvg" in str(type(conv_layer)):
        if _is_depthwise_conv(conv_layer):  # depthwise
            weight_shape = conv_layer.weight.shape  # o, 1, k, k
            if isinstance(conv_layer, Conv2dAvg): # Nếu là conv2davg
                this_num_weight += conv_layer.in_channels * 1 * 1
            else: # normal conv2d
                this_num_weight += conv_layer.in_channels * weight_shape[2] * weight_shape[3]
        elif isinstance(conv_layer, Conv2dAvg): # nếu là Conv2dAvg mà không phải depthwise
            weight_shape = conv_layer.weight.shape
            this_num_weight += weight_shape[0] * weight_shape[1] * 1 * 1 # Bỏ 2 dimension sau vì cái lớp này tính sum của ma trận weight
        else: # Không depthwise lẫn Conv2dAvg
            weight_shape = conv_layer.weight.shape
            if conv_layer.groups == 1:  # normal conv
                this_num_weight += (weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3])
            else:  # group conv (lite residual)
                this_num_weight += conv_layer.weight.data.numel() # Not sure
                
    return str(round(this_num_weight*element_size/(1024*1024), 2)) + " MB"

def register_hook_for_conv(model, hook_size, consider_active_only=False, freeze_cfgs=None):
    '''
    model: Mô hình
    hook_size: hook lưu lại input/output size tại mỗi lớp convolution của model
    consider_active_only: True - Chỉ consider các lớp active khi finetune | False - Consider tất
    freeze_cfgs: cấu hình xem các lớp nào bị freeze, được định nghĩa trong folder trung_configs

    => Hàm này đăng kí hook_size cho model để lưu lại input/output size tại mỗi lớp convolution
    '''
    if not consider_active_only:
        conv_layers = get_all_conv(model)
    else:
        if freeze_cfgs == None:
            conv_layers = get_all_conv(model)
        else:
            conv_layers = get_active_conv(model, freeze_cfgs)

    for conv_layer in  conv_layers:
        conv_layer.register_forward_hook(hook_size)


# def get_activation_size(model, hook_size, radius, consider_active_only=False, freeze_cfgs=None, element_size=4, unit="MB"): # element_size = 4 bytes
#     if not consider_active_only:
#         conv2d_layers = get_all_conv(model)
#     else:
#         if freeze_cfgs == None:
#             conv2d_layers = get_all_conv(model)
#         else:
#             conv2d_layers = get_active_conv(model, freeze_cfgs)

#     input_sizes = hook_size.input_size
#     output_sizes = hook_size.output_size
#     num_element = 0
#     idx = 0
#     for key, input_size in input_sizes.items():
#         input_size = th.tensor(input_size)
#         stride = conv2d_layers[idx].stride
#         x_h, x_w = input_size[-2:]
#         h, w = output_sizes[key][-2:]

#         if isinstance(key, Conv2dAvg): # Nếu key là Conv2dAVG    
#             p_h, p_w = ceil(h / radius), ceil(w / radius)
#             x_order_h, x_order_w = radius * stride[0], radius * stride[1]
#             x_pad_h, x_pad_w = ceil((p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)

#             x_sum_height = ((x_h + 2 * x_pad_h - x_order_h) // x_order_h) + 1
#             x_sum_width = ((x_w + 2 * x_pad_w - x_order_w) // x_order_w) + 1

#             num_element += int(1 * input_size[1] * x_sum_height * x_sum_width) # Bỏ qua số batch

#         elif isinstance(key, nn.modules.conv.Conv2d): # Nếu key là Conv2d
#             # padding_size = key.padding
#             # kernel_size = key.kernel_size
#             # x_height = ((x_h + 2 * padding_size[0] - kernel_size[0]) // stride[0]) + 1
#             # x_width = ((x_w + 2 * padding_size[1] - kernel_size[1]) // stride[1]) + 1
#             # num_element += int(1 * input_size[1] * x_height * x_width) # Bỏ qua số batch
#             num_element += int(1 * input_size[1] * input_size[2] * input_size[3]) # Bỏ qua số batch, Lưu luôn như này do trong hàm forward của convavg, nó lưu thẳng x sau forward
#         idx += 1
#     if unit == "MB":
#         return str(round(num_element*element_size/(1024*1024), 2)) + " MB"
#     elif unit == "KB":
#         return str(round(num_element*element_size/(1024), 2)) + " KB"
#     else:
#         raise ValueError("Unit is not suitable")
