# Bản của mình
import os
import numpy as np
import torch as th
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from custom_op.register import register_filter, register_SVD_with_var, register_conv_batch, register_HOSVD_with_var, register_GKPD_HOSVD
from custom_op.conv_avg import Conv2dAvg
from custom_op.conv_svd_with_var import Conv2dSVD_with_var
from custom_op.conv_avg_batch import Conv2d_Avg_Batch
from custom_op.conv_hosvd_with_var import Conv2dHOSVD_with_var
from util import freeze_layers, get_total_weight_size, get_all_conv_with_name, attach_hooks_for_conv
from math import ceil
from models.encoders import get_encoder
from functools import reduce
import logging

class ClassificationModel(LightningModule):
    def __init__(self, backbone: str, backbone_args, num_classes,
                 learning_rate, weight_decay, set_bn_eval, load = None,

                 num_of_finetune=None, 
                 with_avg_batch=False, with_HOSVD_with_var_compression = False, with_SVD_with_var_compression=False, with_grad_filter=False, with_gkpd_hosvd=False,
                 SVD_var=None, filt_radius=None,
                 
                 use_sgd=False, momentum=0.9, anneling_steps=8008, scheduler_interval='step',
                 lr_warmup=0, init_lr_prod=0.25):
        super(ClassificationModel, self).__init__()
        self.backbone_name = backbone
        self.backbone = get_encoder(backbone, **backbone_args) # Nếu weights (trong backbone_args) được định nghĩa (ssl hoặc sswl) thì load weight từ online về (trong models/encoders/resnet.py hoặc mcunet.py hoặc mobilenet.py)        
        if backbone != "swinT":
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(self.backbone._out_channels[-1], num_classes)
        else:
            self.backbone.head = nn.Linear(in_features=768, out_features=num_classes, bias=True) # Thay lớp cuối của swinT
            # self.classifier = nn.Linear(in_features=768, out_features=num_classes, bias=True) # Thay lớp cuối của swinT
            
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.set_bn_eval = set_bn_eval
        self.acc = Accuracy(num_classes=num_classes)
        ##
        self.with_avg_batch = with_avg_batch
        self.with_HOSVD_with_var_compression = with_HOSVD_with_var_compression
        self.with_gkpd_hosvd = with_gkpd_hosvd
        self.with_SVD_with_var_compression = with_SVD_with_var_compression
        self.with_grad_filter = with_grad_filter
        self.SVD_var = SVD_var
        self.filt_radius = filt_radius
        self.use_sgd = use_sgd
        self.momentum = momentum
        self.anneling_steps = anneling_steps
        self.scheduler_interval = scheduler_interval
        self.lr_warmup = lr_warmup
        self.init_lr_prod = init_lr_prod
        self.hook = {} # Hook being a dict: where key is the module name and value is the hook
        self.num_of_finetune = num_of_finetune

        # self.num_of_validation_batch = 0
        

        self.svd_size = []

        self.k0_hosvd = []
        self.k1_hosvd = []
        self.k2_hosvd = []
        self.k3_hosvd = []
        self.raw_size = []
        self.k_hosvd = [self.k0_hosvd, self.k1_hosvd, self.k2_hosvd, self.k3_hosvd, self.raw_size] # mỗi phần tử của list này là 1 list có độ dài bằng số trained batch * num_of_finetune, vì mỗi batch sẽ có k khác nhau.

        self.k1_gkpd_hosvd_1 = []
        self.k2_gkpd_hosvd_1 = []
        self.k3_gkpd_hosvd_1 = []
        self.k4_gkpd_hosvd_1 = []
        self.raw_size_1 = []
        self.k_gkpd_hosvd_1 = [self.k1_gkpd_hosvd_1, 
                                self.k2_gkpd_hosvd_1, 
                                self.k3_gkpd_hosvd_1, 
                                self.k4_gkpd_hosvd_1,
                                self.raw_size_1]

        self.k1_gkpd_hosvd_2 = []
        self.k2_gkpd_hosvd_2 = []
        self.k3_gkpd_hosvd_2 = []
        self.k4_gkpd_hosvd_2 = []
        self.raw_size_2 = []
        self.k_gkpd_hosvd_2 = [self.k1_gkpd_hosvd_2, 
                                self.k2_gkpd_hosvd_2, 
                                self.k3_gkpd_hosvd_2, 
                                self.k4_gkpd_hosvd_2,
                                self.raw_size_2]
        #############################################################
        filter_cfgs_ = get_all_conv_with_name(self) # Lấy 1 dict bao gồm tên và các lớp conv2d
        filter_cfgs = {}

        if num_of_finetune == "all": # Nếu finetune toàn bộ
            self.num_of_finetune = len(filter_cfgs_)
            if with_SVD_with_var_compression:
                filter_cfgs = {"SVD_var": SVD_var, "svd_size": self.svd_size}
            elif with_HOSVD_with_var_compression:
                filter_cfgs = {"SVD_var": SVD_var, "k_hosvd": self.k_hosvd}
            elif with_gkpd_hosvd:
                filter_cfgs = {"SVD_var": SVD_var, "k_gkpd_hosvd_1": self.k_gkpd_hosvd_1, "k_gkpd_hosvd_2": self.k_gkpd_hosvd_2}
            elif with_grad_filter:
                filter_cfgs = {"radius": filt_radius}

            filter_cfgs["cfgs"] = filter_cfgs_
            filter_cfgs["type"] = "conv"

        elif num_of_finetune > len(filter_cfgs_): # Nếu finetune toàn bộ
            self.num_of_finetune = len(filter_cfgs_)
            logging.info("[Warning] number of finetuned layers is bigger than the total number of conv layers in the network => Finetune all the network")
            if with_SVD_with_var_compression:
                filter_cfgs = {"SVD_var": SVD_var, "svd_size": self.svd_size}
            elif with_HOSVD_with_var_compression:
                filter_cfgs = {"SVD_var": SVD_var, "k_hosvd": self.k_hosvd}
            elif with_gkpd_hosvd:
                filter_cfgs = {"SVD_var": SVD_var, "k_gkpd_hosvd_1": self.k_gkpd_hosvd_1, "k_gkpd_hosvd_2": self.k_gkpd_hosvd_2}
            elif with_grad_filter:
                filter_cfgs = {"radius": filt_radius}

            filter_cfgs["cfgs"] = filter_cfgs_
            filter_cfgs["type"] = "conv"

        elif num_of_finetune is not None and num_of_finetune != 0 and num_of_finetune != "all": # Nếu có finetune nhưng không phải toàn bộ
            filter_cfgs_ = dict(list(filter_cfgs_.items())[-num_of_finetune:]) # Chỉ áp dụng filter vào num_of_finetune conv2d layer cuối
            for name, mod in self.named_modules():
                if len(list(mod.children())) == 0 and name not in filter_cfgs_.keys() and name != '':
                    mod.eval()
                    for param in mod.parameters():
                        param.requires_grad = False # Freeze layer
                elif name in filter_cfgs_.keys(): # Khi duyệt đến layer sẽ được finetune => break. Vì đằng sau các conv2d layer này còn có thể có các lớp fc, gì gì đó mà vẫn cần finetune
                    break
            if with_SVD_with_var_compression:
                filter_cfgs = {"SVD_var": SVD_var, "svd_size": self.svd_size}
            elif with_HOSVD_with_var_compression:
                filter_cfgs = {"SVD_var": SVD_var, "k_hosvd": self.k_hosvd}
            elif with_gkpd_hosvd:
                filter_cfgs = {"SVD_var": SVD_var, "k_gkpd_hosvd_1": self.k_gkpd_hosvd_1, "k_gkpd_hosvd_2": self.k_gkpd_hosvd_2}
            elif with_grad_filter:
                filter_cfgs = {"radius": filt_radius}
            filter_cfgs["cfgs"] = filter_cfgs_
            filter_cfgs["type"] = "conv"
        
        elif num_of_finetune == 0 or num_of_finetune == None: # Nếu không finetune thì freeze all
            for name, mod in self.named_modules():
                if name != '':
                    path_seq = name.split('.')
                    target = reduce(getattr, path_seq, self)
                    target.eval()
                    for param in target.parameters():
                        param.requires_grad = False # Freeze layer
            filter_cfgs = -1
        else:
            filter_cfgs = -1
        #######################################################################
        

        if load != None:
            state_dict = th.load(load)['state_dict']
            self.load_state_dict(state_dict)
            
        # logging.info(f"Before registering filter: Weight size: {get_total_weight_size(self)}")
        if with_HOSVD_with_var_compression:
            register_HOSVD_with_var(self, filter_cfgs)
        elif with_avg_batch:
            register_conv_batch(self, filter_cfgs)
        elif with_SVD_with_var_compression:
            register_SVD_with_var(self, filter_cfgs)
            # logging.info(f"After registering SVD with var compression: Weight size: {get_total_weight_size(self)}")
        elif with_grad_filter:
            register_filter(self, filter_cfgs)
            # logging.info(f"After registering gradient filter: Weight size: {get_total_weight_size(self)}")
        elif with_gkpd_hosvd:
            register_GKPD_HOSVD(self, filter_cfgs)


        self.acc.reset()
    def activate_hooks(self, is_activated=True):
        for h in self.hook:
            self.hook[h].activate(is_activated)

    def remove_hooks(self):
        for h in self.hook:
            self.hook[h].remove()
        logging.info("Hook is removed")
    def reset_svd_size(self):
        self.svd_size.clear()    
    def reset_k_hosvd(self):
        self.k0_hosvd.clear()
        self.k1_hosvd.clear()
        self.k2_hosvd.clear()
        self.k3_hosvd.clear()
        self.raw_size.clear()
    def reset_k_gkpd_hosvd(self):
        self.k1_gkpd_hosvd_1.clear()
        self.k2_gkpd_hosvd_1.clear()
        self.k3_gkpd_hosvd_1.clear()
        self.k4_gkpd_hosvd_1.clear()
        self.raw_size_1.clear()
        
        self.k1_gkpd_hosvd_2.clear()
        self.k2_gkpd_hosvd_2.clear()
        self.k3_gkpd_hosvd_2.clear()
        self.k4_gkpd_hosvd_2.clear()
        self.raw_size_2.clear()

    def get_activation_size(self, trainer, data, consider_active_only=False, element_size=4, unit="MB"): # element_size = 4 bytes
        # Register hook to log input/output size
        train_data_size = th.tensor([data.batch_size, 3, data.width, data.height]) # 3 channels
        attach_hooks_for_conv(self, consider_active_only=consider_active_only)
        self.activate_hooks(True)
        #############################################################################
        _, first_hook = next(iter(self.hook.items()))
        if first_hook.active:
            logging.info("Hook is activated")
        else:
            logging.info("[Warning] Hook is not activated !!")
        #############################################################################
        # Create a temporary input for model so the hook can record input/output size
        if isinstance(first_hook.module, Conv2dSVD_with_var) or isinstance(first_hook.module, Conv2dHOSVD_with_var):
            trainer.validate(self, datamodule=data)  
        elif isinstance(first_hook.module, Conv2dAvg) or isinstance(first_hook.module, Conv2d_Avg_Batch)\
            or isinstance(first_hook.module, nn.modules.conv.Conv2d):
            _ = th.rand(train_data_size[0], train_data_size[1], train_data_size[2], train_data_size[3])
            _ = self(_)

        num_element = 0
        for name in self.hook:
            print(name)
            input_size = th.tensor(self.hook[name].input_size).clone().detach()
            stride = self.hook[name].module.stride
            x_h, x_w = input_size[-2:]
            h, w = self.hook[name].output_size[-2:]

            if isinstance(self.hook[name].module, Conv2dAvg): 
                p_h, p_w = ceil(h / self.filt_radius), ceil(w / self.filt_radius)
                x_order_h, x_order_w = self.filt_radius * stride[0], self.filt_radius * stride[1]
                x_pad_h, x_pad_w = ceil((p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)

                x_sum_height = ((x_h + 2 * x_pad_h - x_order_h) // x_order_h) + 1
                x_sum_width = ((x_w + 2 * x_pad_w - x_order_w) // x_order_w) + 1

                num_element += int(input_size[0] * input_size[1] * x_sum_height * x_sum_width)

                # print(name, ": ", self.hook[name].module, " ----- ",int(input_size[0] * input_size[1] * x_sum_height * x_sum_width))

            elif isinstance(self.hook[name].module, Conv2d_Avg_Batch):
                temp_tensor = th.rand(input_size[0], input_size[1], input_size[2], input_size[3], dtype=th.float32)  
                from custom_op.conv_avg_batch import conv_avg_batch
                x = conv_avg_batch(temp_tensor)
                num_element += x.numel() + th.tensor(temp_tensor.shape).numel()

            elif isinstance(self.hook[name].module, Conv2dSVD_with_var):
                ############## Kiểu reshape activation map theo dim
                from custom_op.conv_svd_with_var import truncated_svd
                num_element_all = 0
                for input in self.hook[name].inputs:
                    Uk_Sk, Vk_t = truncated_svd(input, var=self.SVD_var)
                    num_element_all += Uk_Sk.numel() + Vk_t.numel()
                num_element += num_element_all/len(self.hook[name].inputs)
            
            elif isinstance(self.hook[name].module, Conv2dHOSVD_with_var):
                ############## Kiểu reshape activation map theo dim
                from custom_op.conv_hosvd_with_var import hosvd
                num_element_all = 0
                for input in self.hook[name].inputs:
                    S, u0, u1, u2, u3, _ = hosvd(input, var=self.SVD_var)
                    num_element_all += S.numel() + u0.numel() + u1.numel() + u2.numel() + u3.numel()
                    # print(name, ": ", input.shape, " ----- ", S.numel() + u0.numel() + u1.numel() + u2.numel() + u3.numel(), " --- ")
                num_element += num_element_all/len(self.hook[name].inputs)
            
            elif isinstance(self.hook[name].module, nn.modules.conv.Conv2d):
                num_element += int(input_size[0] * input_size[1] * input_size[2] * input_size[3])

        self.remove_hooks()
        if self.with_HOSVD_with_var_compression:
            filename = self.backbone_name + "_HOSVD_" + str(self.SVD_var)
        elif self.with_avg_batch:
            filename = self.backbone_name + "_avg_batch"
        elif self.with_grad_filter:
            filename = self.backbone_name + "_gradfilt_r" + str(self.filt_radius)
        elif self.with_SVD_with_var_compression:
            filename = self.backbone_name + "_SVD_" + str(self.SVD_var)
        else:
            filename = self.backbone_name + "_base"

        def write_to_txt(filename, content):
            if not os.path.exists("mem_res/"):
                os.makedirs("mem_res/")
            filename = "mem_res/" + filename + ".txt"
            try:
                with open(filename, 'a') as file:
                    file.write(content + '\n')
            except FileNotFoundError:
                with open(filename, 'w') as file:
                    file.write(content + '\n')

        if unit == "Byte":
            res = str(num_element*element_size)
            write_to_txt(filename, res)
            return str(num_element*element_size) + " Bytes"
        if unit == "MB":
            res = str((num_element*element_size)/(1024*1024))
            write_to_txt(filename, res)
            return res + " MB"
        elif unit == "KB":
            res = str((num_element*element_size)/(1024))
            write_to_txt(filename, res)
            return res + " KB"
        else:
            raise ValueError("Unit is not suitable")
        


    # def compute_Conv2d_flops(self, trainer, data, consider_active_only=False, element_size=4, unit="MB"): # element_size = 4 bytes
    #     # Register hook to log input/output size
    #     train_data_size = th.tensor([data.batch_size, 3, data.width, data.height]) # 3 channels
    #     attach_hooks_for_conv(self, consider_active_only=consider_active_only)
    #     self.activate_hooks(True)
    #     #############################################################################
    #     _, first_hook = next(iter(self.hook.items()))
    #     if first_hook.active:
    #         logging.info("Hook is activated")
    #     else:
    #         logging.info("[Warning] Hook is not activated !!")
    #     #############################################################################
    #     # Create a temporary input for model so the hook can record input/output size
    #     if isinstance(first_hook.module, Conv2dSVD_with_var) or isinstance(first_hook.module, Conv2dHOSVD_with_var):
    #         trainer.validate(self, datamodule=data)  
    #     elif isinstance(first_hook.module, Conv2dAvg) or isinstance(first_hook.module, Conv2d_Avg_Batch)\
    #         or isinstance(first_hook.module, Conv2dSVD) or isinstance(first_hook.module, nn.modules.conv.Conv2d):
    #         _ = th.rand(train_data_size[0], train_data_size[1], train_data_size[2], train_data_size[3])
    #         _ = self(_)

    #     FLOPs = 0
    #     for name in self.hook:
    #         Bx, Cx, Hx, Wx = th.tensor(self.hook[name].input_size).clone().detach() # Batch size, Số channel, cao, rộng của input
    #         By, Cy, Hy, Wy = self.hook[name].output_size # Batch size, Số channel, cao, rộng của output
    #         Hk, Wk = self.hook[name].module.kernel_size # Kích thước kernel
    #         if isinstance(self.hook[name].module, Conv2dAvg): # Nếu key là Conv2dAVG    
    #             flops = float(ceil(Hy / self.filt_radius) * ceil(Wy / self.filt_radius) * Cx * (2 * Cy - 1)) # Công thức (15) trong paper) # Đây là FLOPs của quá trình tính gradient trên input
    #             backprop_overhead = float(((self.filt_radius**2 - 1) / (2 * Cx)) * flops) # ratio of backwarđ propagation overhead
    #             flops += backprop_overhead
                
    #             # from math import floor
    #             # stride = self.hook[name].module.stride
    #             # x_h, x_w = Hx, Wx
    #             # k_h, k_w = Hk, Wk
    #             # h, w = Hy, Wy
    #             # p_h, p_w = ceil(h / self.filt_radius), ceil(w / self.filt_radius)
    #             # x_order_h, x_order_w = self.filt_radius * stride[0], self.filt_radius * stride[1]
    #             # x_pad_h, x_pad_w = ceil((p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)
                
    #             # x_sum_height = ((x_h + 2 * x_pad_h - x_order_h) // x_order_h) + 1
    #             # x_sum_width = ((x_w + 2 * x_pad_w - x_order_w) // x_order_w) + 1


    #             # p_h, p_w = x_sum_height, x_sum_width
    #             # _, c_out, gy_h, gy_w = self.hook[name].output_size
    #             # grad_y_pad_h, grad_y_pad_w = ceil((p_h * self.filt_radius - gy_h) / 2), ceil((p_w * self.filt_radius - gy_w) / 2)
    #             # grad_y_avg_shape = [self.hook[name].output_size[0],
    #             #                     self.hook[name].output_size[1],
    #             #                     floor((self.hook[name].output_size[2] + 2 * grad_y_pad_h - self.filt_radius) / self.filt_radius + 1), 
    #             #                     floor((self.hook[name].output_size[3] + 2 * grad_y_pad_w - self.filt_radius) / self.filt_radius + 1)] # dựa trên công thức ở https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d

    #             # if self.hook[name].module.groups != 1:
    #             #     a1, a2, a3, a4 = Bx, Cx, x_sum_height, x_sum_width
    #             #     # b1, b2, b3, b4 = grad_y_avg_shape
    #             #     mul_flops = a1*a2*a3*a4 # x_sum * grad_y_avg do đây là phép element wise
    #             #     add_flops = (a2*a3-1)*a1 + a1-1 # .sum(dim=(0, 2, 3))

    #             #     flops = add_flops + mul_flops
    #             # else:
    #             #     gy_shape_1, gy_shape_2 = grad_y_avg_shape[1], grad_y_avg_shape[0] * grad_y_avg_shape[2] * grad_y_avg_shape[3]
    #             #     gx_shape_1, gx_shape_2 = Bx * x_sum_height * x_sum_width, Cx
    #             #     flops = gy_shape_1 * gy_shape_2 * (2 * gx_shape_2 - 1) # gy @ gx
    #             # print("flops: ", flops)
    #             FLOPs += flops
    #         elif isinstance(self.hook[name].module, Conv2dSVD_with_var): # SVD case
    #             ############## Kiểu reshape activation map theo dim
    #             from custom_op.conv_svd_with_var import truncated_svd
    #             backprop_overhead = 0
    #             for input in self.hook[name].inputs: # Duyệt qua tất cả các batch of input tại mỗi layer để tính tổng tất cả các phần tử phải lưu của tất cả chúng tại mỗi layer
    #                 Uk_Sk, Vk_t = truncated_svd(input, var=self.SVD_var) # Tính cả số phần tử trong 1 batch
    #                 backprop_overhead += 2*Uk_Sk.shape[0]*Uk_Sk.shape[1]*Vk_t.shape[1]  # Số lượng FLOPs để restore tensor
    #             flops = int(2 * Cx * Cy * Wy * Hy * Wk * Hk)
    #             backprop_overhead = backprop_overhead / len(self.hook[name].inputs)
    #             flops += backprop_overhead

    #             FLOPs += flops
            
    #         elif isinstance(self.hook[name].module, nn.modules.conv.Conv2d): # Nếu key là Conv2d
    #             FLOPs += int(2 * Cx * Cy * Wy * Hy * Wk * Hk) # Công thức (1) trong paper để tính gradient của input + FLOPs để tính Frobenius Inner Product để tính gradient của kernel (Có thể dùng 2*Hy*Wy) => Nói chung là kích thước x, y ở đây bằng nhau
    #     self.remove_hooks()

    #     if unit == "MB":
    #         return str(round(float(FLOPs/(1024*1024)), 2)) + " MB"
    #     elif unit == "KB":
    #         return str(round(float(FLOPs/(1024)), 2)) + " KB"
    #     else:
    #         raise ValueError("Unit is not suitable")

    def freeze_layers(self):
        freeze_layers(self, self.freeze_cfgs)

    def configure_optimizers(self):
        if self.use_sgd:
            optimizer = th.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
            if self.lr_warmup == 0:
                scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, self.anneling_steps, eta_min=0.1 * self.learning_rate)
            else:
                def _lr_fn(epoch):
                    if epoch < self.lr_warmup:
                        lr = self.init_lr_prod + (1 - self.init_lr_prod) / (self.lr_warmup - 1) * epoch
                    else:
                        e = epoch - self.lr_warmup
                        es = self.anneling_steps - self.lr_warmup
                        lr = 0.5 * (1 + np.cos(np.pi * e / es))
                    return lr
                scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_fn)
            sch = {
                "scheduler": scheduler,
                'interval': self.scheduler_interval,
                'frequency': 1
            }
            return [optimizer], [sch]
        optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                  lr=self.learning_rate, weight_decay=self.weight_decay, betas=(0.8, 0.9))
        return [optimizer]

    def bn_eval(self):
        def f(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
            m.momentum = 1.0
        self.apply(f)

    def forward(self, x):
        if self.backbone_name != "swinT":
            feat = self.backbone(x)[-1]
            feat = self.pooling(feat)
            feat = feat.flatten(start_dim=1)
            logit = self.classifier(feat)
            return logit
        else:
            logit = self.backbone(x)
            # logit = self.classifier(feat)
            return logit

    def training_step(self, train_batch, batch_idx):
        if self.with_HOSVD_with_var_compression:
            self.reset_k_hosvd() # Reset logged k which is used for logging memory for HOSVD
        if self.with_SVD_with_var_compression:
            self.reset_svd_size()
        if self.with_gkpd_hosvd:
            self.reset_k_gkpd_hosvd()

        if self.set_bn_eval:
            self.bn_eval()
        img, label = train_batch['image'], train_batch['label']
        if img.shape[1] == 1:
            img = th.cat([img] * 3, dim=1)
        logits = self.forward(img)
        pred_cls = th.argmax(logits, dim=-1)
        acc = th.sum(pred_cls == label) / label.shape[0]
        loss = self.loss(logits, label)
        self.log("Train/Loss", loss)
        self.log("Train/Acc", acc)
        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs):

        # if self.with_SVD_with_var_compression:
        #     device = th.device("cuda" if th.cuda.is_available() else "cpu")
        #     svd_size_tensor= th.stack(self.svd_size).t().float() # Chiều 1: 3 kích thước của các thành phần; chiều 2 từng batch của tất cả layer
        #     svd_size_tensor = svd_size_tensor.view(3, -1, self.num_of_finetune) # Shape: 3 kích thước của các thành phần, số batch, num_of_finetune
        #     svd_size_tensor = svd_size_tensor.permute(2, 1, 0) # Shape: num_of_finetune, số batch, 3 kích thước của các thành phần
        #     svd_size_tensor = svd_size_tensor[:, :-self.num_of_validation_batch, :]
        #     # Tính trung bình của mỗi layer
        #     num_element_all = th.mean(svd_size_tensor[:, :, 0] * svd_size_tensor[:, :, 1] + svd_size_tensor[:, :, 1] * svd_size_tensor[:, :, 2], dim=1)
        #     # Tổng hợp các giá trị trung bình của mỗi layer
        #     num_element = th.sum(num_element_all)

            
        #     mem = (num_element*4)#/(1024*1024)
        #     with open(os.path.join(self.logger.log_dir, "activation_memory_Byte.log"), "a") as file:
        #         file.write(str(self.current_epoch) + "\t" + str(float(mem)) + "\n")
        #     self.reset_svd_size()
            
            
        # # Log activation memory at each epoch for HOSVD
        # if self.with_HOSVD_with_var_compression:
        #     device = th.device("cuda" if th.cuda.is_available() else "cpu")

        #     # New way (optimized)
        #     k_hosvd_tensor = th.tensor(self.k_hosvd[:4], device=device).float() # Chiều 1: dimension (4 cái k); chiều 2: k của từng batch của tất cả layer
        #     k_hosvd_tensor = k_hosvd_tensor.view(4, -1, self.num_of_finetune) # Shape: 4 cái k, số batch, num_of_finetune
        #     k_hosvd_tensor = k_hosvd_tensor.permute(2, 1, 0) # Shape: num_of_finetune, số batch, 4 cái k
        #     k_hosvd_tensor = k_hosvd_tensor[:, :-self.num_of_validation_batch, :]

        #     raw_shapes = th.tensor(self.k_hosvd[4], device=device).reshape(-1, self.num_of_finetune, 4) # Shape: num of batch, num_of_finetune, 4 chiều
        #     raw_shapes = raw_shapes.permute(1, 0, 2) # Shape: num_of_finetune, num of batch, 4 chiều
        #     raw_shapes = raw_shapes[:, :-self.num_of_validation_batch, :]
        #     '''
        #     Duyệt theo từng layer (chiều 1: num_of_finetune) -> Duyệt theo từng batch (chiều 2: số batch), tính số phần tử ở đây rồi suy ra trung bình số phần tử tại mỗi batch tại mỗi layer
        #     -> Cộng tất cả ra trung bình số phần tử tại mỗi batch của các layer

        #     '''
        #     num_element_all = th.sum(
        #         k_hosvd_tensor[:, :, 0] * k_hosvd_tensor[:, :, 1] * k_hosvd_tensor[:, :, 2] * k_hosvd_tensor[:, :, 3]
        #         + k_hosvd_tensor[:, :, 0] * raw_shapes[:, :, 0]
        #         + k_hosvd_tensor[:, :, 1] * raw_shapes[:, :, 1]
        #         + k_hosvd_tensor[:, :, 2] * raw_shapes[:, :, 2]
        #         + k_hosvd_tensor[:, :, 3] * raw_shapes[:, :, 3],
        #         dim=1
        #     )
        #     num_element = th.sum(num_element_all) / k_hosvd_tensor.shape[1]

        #     # # Old way
        #     # k_hosvd_tensor = th.tensor(self.k_hosvd[:4]).float() # Chiều 1: dimension (4 cái k); chiều 2: k của từng batch của tất cả layer
        #     # k_hosvd_tensor = k_hosvd_tensor.t()
        #     # num_element = 0
        #     # for i in range(self.num_of_finetune): # Duyệt qua các layer
        #     #     raw_shape = self.k_hosvd[4][i] # Raw shape của activation map tại mỗi layer, đây là một tensor
        #     #     selected_rows = k_hosvd_tensor[i::self.num_of_finetune] # 4 chuỗi k của một layer
        #     #     num_element_all = 0
        #     #     for row in selected_rows:
        #     #         num_element_all += row[0] * row[1] * row[2] * row[3] \
        #     #         + row[0] * raw_shape[0] + row[1] * raw_shape[1] \
        #     #         + row[2] * raw_shape[2] + row[3] * raw_shape[3]
        #     #     num_element += num_element_all/selected_rows.shape[0]

               
        #     mem = (num_element*4)#/(1024*1024)
        #     with open(os.path.join(self.logger.log_dir, "activation_memory_Byte.log"), "a") as file:
        #         file.write(str(self.current_epoch) + "\t" + str(float(mem)) + "\n")
            
        #     self.reset_k_hosvd() # Reset logged k which is used for logging memory for HOSVD
        #     self.num_of_validation_batch = 0
        
        with open(os.path.join(self.logger.log_dir, 'train_loss.log'), 'a') as f:
            mean_loss = th.stack([o['loss'] for o in outputs]).mean()
            f.write(f"{self.current_epoch} {mean_loss}")
            f.write("\n")

        with open(os.path.join(self.logger.log_dir, 'train_acc.log'), 'a') as f:
            mean_acc = th.stack([o['acc'] for o in outputs]).mean()
            f.write(f"{self.current_epoch} {mean_acc}")
            f.write("\n")

    def validation_step(self, val_batch, batch_idx):
        
        # self.num_of_validation_batch += 1


        img, label = val_batch['image'], val_batch['label']
        if img.shape[1] == 1:
            img = th.cat([img] * 3, dim=1)
        logits = self.forward(img)
        probs = logits.softmax(dim=-1)
        pred = th.argmax(logits, dim=1)
        self.acc(probs, label)
        loss = self.loss(logits, label)
        self.log("Val/Loss", loss)
        return {'pred': pred, 'prob': probs, 'label': label}

    def validation_epoch_end(self, outputs):
        if self.with_SVD_with_var_compression:
            device = th.device("cuda" if th.cuda.is_available() else "cpu")
            svd_size_tensor= th.stack(self.svd_size).t().float() # Chiều 1: 3 kích thước của các thành phần; chiều 2 từng batch của tất cả layer
            svd_size_tensor = svd_size_tensor.view(3, -1, self.num_of_finetune) # Shape: 3 kích thước của các thành phần, số batch, num_of_finetune
            svd_size_tensor = svd_size_tensor.permute(2, 1, 0) # Shape: num_of_finetune, số batch, 3 kích thước của các thành phần

            # Tính trung bình của mỗi layer
            num_element_all = th.mean(svd_size_tensor[:, :, 0] * svd_size_tensor[:, :, 1] + svd_size_tensor[:, :, 1] * svd_size_tensor[:, :, 2], dim=1)
            # Tổng hợp các giá trị trung bình của mỗi layer
            num_element = th.sum(num_element_all)

            
            mem = (num_element*4)#/(1024*1024)
            with open(os.path.join(self.logger.log_dir, "activation_memory_Byte.log"), "a") as file:
                file.write(str(self.current_epoch) + "\t" + str(float(mem)) + "\n")
            
            
        # Log activation memory at each epoch for HOSVD
        if self.with_HOSVD_with_var_compression:
            device = th.device("cuda" if th.cuda.is_available() else "cpu")

            # New way (optimized)
            k_hosvd_tensor = th.tensor(self.k_hosvd[:4], device=device).float() # Chiều 1: dimension (4 cái k); chiều 2: k của từng batch của tất cả layer
            k_hosvd_tensor = k_hosvd_tensor.view(4, -1, self.num_of_finetune) # Shape: 4 cái k, số batch, num_of_finetune
            k_hosvd_tensor = k_hosvd_tensor.permute(2, 1, 0) # Shape: num_of_finetune, số batch, 4 cái k
            

            raw_shapes = th.tensor(self.k_hosvd[4], device=device).reshape(-1, self.num_of_finetune, 4) # Shape: num of batch, num_of_finetune, 4 chiều
            raw_shapes = raw_shapes.permute(1, 0, 2) # Shape: num_of_finetune, num of batch, 4 chiều

            '''
            Duyệt theo từng layer (chiều 1: num_of_finetune) -> Duyệt theo từng batch (chiều 2: số batch), tính số phần tử ở đây rồi suy ra trung bình số phần tử tại mỗi batch tại mỗi layer
            -> Cộng tất cả ra trung bình số phần tử tại mỗi batch của các layer

            '''
            num_element_all = th.sum(
                k_hosvd_tensor[:, :, 0] * k_hosvd_tensor[:, :, 1] * k_hosvd_tensor[:, :, 2] * k_hosvd_tensor[:, :, 3]
                + k_hosvd_tensor[:, :, 0] * raw_shapes[:, :, 0]
                + k_hosvd_tensor[:, :, 1] * raw_shapes[:, :, 1]
                + k_hosvd_tensor[:, :, 2] * raw_shapes[:, :, 2]
                + k_hosvd_tensor[:, :, 3] * raw_shapes[:, :, 3],
                dim=1
            )
            num_element = th.sum(num_element_all) / k_hosvd_tensor.shape[1]

            # # Old way
            # k_hosvd_tensor = th.tensor(self.k_hosvd[:4]).float() # Chiều 1: dimension (4 cái k); chiều 2: k của từng batch của tất cả layer
            # k_hosvd_tensor = k_hosvd_tensor.t()
            # num_element = 0
            # for i in range(self.num_of_finetune): # Duyệt qua các layer
            #     raw_shape = self.k_hosvd[4][i] # Raw shape của activation map tại mỗi layer, đây là một tensor
            #     selected_rows = k_hosvd_tensor[i::self.num_of_finetune] # 4 chuỗi k của một layer
            #     num_element_all = 0
            #     for row in selected_rows:
            #         num_element_all += row[0] * row[1] * row[2] * row[3] \
            #         + row[0] * raw_shape[0] + row[1] * raw_shape[1] \
            #         + row[2] * raw_shape[2] + row[3] * raw_shape[3]
            #     num_element += num_element_all/selected_rows.shape[0]

               
            mem = (num_element*4)#/(1024*1024)
            with open(os.path.join(self.logger.log_dir, "activation_memory_Byte.log"), "a") as file:
                file.write(str(self.current_epoch) + "\t" + str(float(mem)) + "\n")
        
        if self.with_gkpd_hosvd:
            device = th.device("cuda" if th.cuda.is_available() else "cpu")

            # New way (optimized)
            k_gkpd_hosvd_1_tensor = th.tensor(self.k_gkpd_hosvd_1[:4], device=device).float() # Chiều 1: dimension (4 cái k); chiều 2: k của từng batch của tất cả layer
            k_gkpd_hosvd_1_tensor = k_gkpd_hosvd_1_tensor.view(4, -1, self.num_of_finetune) # Shape: 4 cái k, số batch, num_of_finetune
            k_gkpd_hosvd_1_tensor = k_gkpd_hosvd_1_tensor.permute(2, 1, 0) # Shape: num_of_finetune, số batch, 4 cái k
            

            raw_shapes1 = th.tensor(self.k_gkpd_hosvd_1[4], device=device).reshape(-1, self.num_of_finetune, 5) # Shape: num of batch, num_of_finetune, 5 chiều
            raw_shapes1 = raw_shapes1.permute(1, 0, 2) # Shape: num_of_finetune, num of batch, 5 chiều

            '''
            Duyệt theo từng layer (chiều 1: num_of_finetune) -> Duyệt theo từng batch (chiều 2: số batch), tính số phần tử ở đây rồi suy ra trung bình số phần tử tại mỗi batch tại mỗi layer
            -> Cộng tất cả ra trung bình số phần tử tại mỗi batch của các layer

            '''
            num_element_all1 = th.sum(
                k_gkpd_hosvd_1_tensor[:, :, 0] * k_gkpd_hosvd_1_tensor[:, :, 1] * k_gkpd_hosvd_1_tensor[:, :, 2] * k_gkpd_hosvd_1_tensor[:, :, 3]
                + k_gkpd_hosvd_1_tensor[:, :, 0] * raw_shapes1[:, :, 1]
                + k_gkpd_hosvd_1_tensor[:, :, 1] * raw_shapes1[:, :, 2]
                + k_gkpd_hosvd_1_tensor[:, :, 2] * raw_shapes1[:, :, 3]
                + k_gkpd_hosvd_1_tensor[:, :, 3] * raw_shapes1[:, :, 4]
                + raw_shapes1[:, :, 0],
                dim=1
            )
            num_element1 = th.sum(num_element_all1) / k_gkpd_hosvd_1_tensor.shape[1]

            k_gkpd_hosvd_2_tensor = th.tensor(self.k_gkpd_hosvd_2[:4], device=device).float() # Chiều 1: dimension (4 cái k); chiều 2: k của từng batch của tất cả layer
            k_gkpd_hosvd_2_tensor = k_gkpd_hosvd_2_tensor.view(4, -1, self.num_of_finetune) # Shape: 4 cái k, số batch, num_of_finetune
            k_gkpd_hosvd_2_tensor = k_gkpd_hosvd_2_tensor.permute(2, 1, 0) # Shape: num_of_finetune, số batch, 4 cái k
            

            raw_shapes2 = th.tensor(self.k_gkpd_hosvd_2[4], device=device).reshape(-1, self.num_of_finetune, 5) # Shape: num of batch, num_of_finetune, 5 chiều
            raw_shapes2 = raw_shapes2.permute(1, 0, 2) # Shape: num_of_finetune, num of batch, 4 chiều

            '''
            Duyệt theo từng layer (chiều 1: num_of_finetune) -> Duyệt theo từng batch (chiều 2: số batch), tính số phần tử ở đây rồi suy ra trung bình số phần tử tại mỗi batch tại mỗi layer
            -> Cộng tất cả ra trung bình số phần tử tại mỗi batch của các layer

            '''
            num_element_all2 = th.sum(
                k_gkpd_hosvd_2_tensor[:, :, 0] * k_gkpd_hosvd_2_tensor[:, :, 1] * k_gkpd_hosvd_2_tensor[:, :, 2] * k_gkpd_hosvd_2_tensor[:, :, 3]
                + k_gkpd_hosvd_2_tensor[:, :, 0] * raw_shapes2[:, :, 1]
                + k_gkpd_hosvd_2_tensor[:, :, 1] * raw_shapes2[:, :, 2]
                + k_gkpd_hosvd_2_tensor[:, :, 2] * raw_shapes2[:, :, 3]
                + k_gkpd_hosvd_2_tensor[:, :, 3] * raw_shapes2[:, :, 4]
                + raw_shapes2[:, :, 0],
                dim=1
            )
            num_element2 = th.sum(num_element_all2) / k_gkpd_hosvd_2_tensor.shape[1]

               
            mem = (num_element1+num_element2)*4
            with open(os.path.join(self.logger.log_dir, "activation_memory_Byte.log"), "a") as file:
                file.write(str(self.current_epoch) + "\t" + str(float(mem)) + "\n")

            
        f = open(os.path.join(self.logger.log_dir, 'val.log'), 'a') if self.logger is not None else None
        acc = self.acc.compute()
        if self.logger is not None:
            f.write(f"{self.current_epoch} {acc}\n")
            f.close()
        self.log("Val/Acc", acc)
        self.log("val-acc", acc)
        self.acc.reset()