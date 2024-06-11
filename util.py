from math import ceil
import torch.nn as nn
import re
from custom_op.conv_avg import Conv2dAvg
from custom_op.conv_avg_batch import Conv2d_Avg_Batch
from custom_op.conv_svd_with_var import Conv2dSVD_with_var
from custom_op.conv_hosvd_with_var import Conv2dHOSVD_with_var
from custom_op_linear.linear_hosvd_with_var import Linear_HOSVD
import torch

###################################### Their functions ##############################
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

############################### My functions ###############################################
def get_all_conv(model):
    conv_layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.modules.conv.Conv2d) or isinstance(mod, Conv2dAvg) or isinstance(mod, Conv2d_Avg_Batch): # Nếu mod là Conv2d hoặc Conv2dAvg hoặc Conv2d_Avg_Batch
            conv_layers.append(mod)
    return conv_layers

# def get_active_conv(model, freeze_cfgs):
#     if freeze_cfgs == None:
#         return get_all_conv(model)
#     else:
#         list_freeze_cfgs = []
#         for cfg in freeze_cfgs:
#             path = cfg['path'].replace(" ", ".")
#             if '[' in path or ']' in path:
#                 path = path.replace("[", "").replace("]", "")
#             list_freeze_cfgs.append(path)

#         active_conv_layers = []
#         # active_conv_layers = {}
#         for name, mod in model.named_modules():
#             if isinstance(mod, nn.modules.conv.Conv2d) or isinstance(mod, Conv2dAvg)  or isinstance(mod, Conv2d_Avg_Batch): # Nếu mod là Conv2d hoặc Conv2dAvg hoặc Conv2d_Avg_Batch
#                 if not any(re.match(f"^{prefix}(?:\\.|$)", name) for prefix in list_freeze_cfgs): # Module đang xét không thuộc danh sách freeze
#                     active_conv_layers.append(mod)
#                     # active_conv_layers[name] = mod
#         return active_conv_layers
    

def get_total_weight_size(model, element_size=4): # element_size = 4 bytes
    def _is_depthwise_conv(conv):
        return conv.groups == conv.in_channels == conv.out_channels

    conv_layers = get_all_conv(model)
    this_num_weight = 0
    # print(conv_layers)
    for conv_layer in conv_layers:
        # if "Conv2dAvg" in str(type(conv_layer)):
        if _is_depthwise_conv(conv_layer):  # depthwise
            # print(conv_layer, ":     ", conv_layer.weight.shape)

            weight_shape = conv_layer.weight.shape  # o, 1, k, k
            if isinstance(conv_layer, Conv2dAvg): # Nếu là conv2dAvg
                this_num_weight += conv_layer.in_channels * 1 * 1
            elif isinstance(conv_layer, Conv2d_Avg_Batch):
                this_num_weight += conv_layer.in_channels * weight_shape[2] * weight_shape[3] ############
            else: # normal conv2d
                this_num_weight += conv_layer.in_channels * weight_shape[2] * weight_shape[3]
        elif isinstance(conv_layer, Conv2dAvg): # nếu là Conv2dAvg mà không phải depthwise
            weight_shape = conv_layer.weight.shape
            this_num_weight += weight_shape[0] * weight_shape[1] * 1 * 1 # Bỏ 2 dimension sau vì cái lớp này tính sum của ma trận weight
        elif isinstance(conv_layer, Conv2d_Avg_Batch): ################################
            weight_shape = conv_layer.weight.shape
            if conv_layer.groups == 1:  # normal conv
                this_num_weight += (weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3])
            else:  # group conv (lite residual)
                this_num_weight += conv_layer.weight.data.numel() # Not sure
        else: # Không depthwise lẫn Conv2dAvg
            weight_shape = conv_layer.weight.shape
            if conv_layer.groups == 1:  # normal conv
                this_num_weight += (weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3])
            else:  # group conv (lite residual)
                this_num_weight += conv_layer.weight.data.numel() # Not sure
                
    # return str(round(this_num_weight*element_size/(1024*1024), 2)) + " MB"
    return str(round(this_num_weight*element_size/(1024), 2)) + " KB"


########## Phiên bản hook khác ##########

def get_all_linear_with_name(model):
    linear_layers = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.modules.linear.Linear) or isinstance(mod, Linear_HOSVD):
            linear_layers[name] = mod
    return linear_layers

def get_active_linear_with_name(model):
    total_linear_layer = get_all_linear_with_name(model)
    if model.num_of_finetune == "all" or model.num_of_finetune > len(total_linear_layer):
        return total_linear_layer
    elif model.num_of_finetune == None or model.num_of_finetune == 0:
        return -1 # Không có conv layer nào được finetuned
    else:
        active_linear_layers = dict(list(total_linear_layer.items())[-model.num_of_finetune:]) # Chỉ áp dụng filter vào num_of_finetune conv2d layer cuối
        return active_linear_layers

def get_all_conv_with_name(model):
    conv_layers = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.modules.conv.Conv2d) or isinstance(mod, Conv2dAvg) or isinstance(mod, Conv2d_Avg_Batch) or isinstance(mod, Conv2dSVD_with_var) or isinstance(mod, Conv2dHOSVD_with_var):
            conv_layers[name] = mod
    return conv_layers
    
def get_active_conv_with_name(model):
    total_conv_layer = get_all_conv_with_name(model)
    if model.num_of_finetune == "all" or model.num_of_finetune > len(total_conv_layer):
        return total_conv_layer
    elif model.num_of_finetune == None or model.num_of_finetune == 0:
        return -1 # Không có conv layer nào được finetuned
    else:
        active_conv_layers = dict(list(total_conv_layer.items())[-model.num_of_finetune:]) # Chỉ áp dụng filter vào num_of_finetune conv2d layer cuối
        return active_conv_layers

class Hook: # Lưu lại các input/output size của mô hình
    def __init__(self, module):
        self.module = module
        self.input_size = torch.zeros(4)
        self.output_size = torch.zeros(4)
        
        self.inputs = []#torch.empty(0, 4)
        
        self.active = True
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        if not self.active:
            return
        Input = input[0].clone().detach()
        Output = output.clone().detach()
        self.input_size = Input.shape
        self.output_size = Output.shape

        self.inputs.append(Input)
    def activate(self, active):
        self.active = active
    def remove(self):
        self.active = False
        self.hook.remove()
        # print("Hook is removed")

def attach_hooks_for_conv(model, consider_active_only=False):
    '''
    model: Mô hình
    consider_active_only: True - Chỉ consider các lớp active khi finetune | False - Consider tất
    model.freeze_cfgs: cấu hình xem các lớp nào bị freeze, được định nghĩa trong folder trung_configs

    => Hàm này đăng kí hook cho model để lưu lại input/output size tại mỗi lớp convolution
    '''
    if not consider_active_only:
        conv_layers = get_all_conv_with_name(model)
    else:
        conv_layers = get_active_conv_with_name(model)
    assert conv_layers != -1, "[Warning] Consider activate conv2d only but no conv2d is finetuned => No hook is attached !!"

    for name, mod in  conv_layers.items():
        model.hook[name] = Hook(mod) # attribute hook trong model là dict

def attach_hooks_for_linear(model, consider_active_only=False):
    '''
    model: Mô hình
    consider_active_only: True - Chỉ consider các lớp active khi finetune | False - Consider tất
    model.freeze_cfgs: cấu hình xem các lớp nào bị freeze, được định nghĩa trong folder trung_configs

    => Hàm này đăng kí hook cho model để lưu lại input/output size tại mỗi lớp convolution
    '''
    if not consider_active_only:
        linear_layers = get_all_linear_with_name(model)
    else:
        linear_layers = get_active_linear_with_name(model)
    assert linear_layers != -1, "[Warning] Consider activate conv2d only but no conv2d is finetuned => No hook is attached !!"

    for name, mod in  linear_layers.items():
        model.hook[name] = Hook(mod) # attribute hook trong model là dict