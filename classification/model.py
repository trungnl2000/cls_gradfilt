# Bản của mình
import os
import numpy as np
import torch as th
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from custom_op.register import register_filter
from custom_op.conv_avg import Conv2dAvg
from custom_op.conv_svd import Conv2dSVD
from util import freeze_layers, get_total_weight_size, Conv2dSizeHook, register_hook_for_conv, get_all_conv, get_active_conv, get_all_conv_with_name, get_active_conv_with_name, attach_hooks_for_conv, Hook
from math import ceil
from models.encoders import get_encoder
from functools import reduce
import logging


class ClassificationModel(LightningModule):
    # def __init__(self, backbone: str, backbone_args, num_classes, learning_rate, weight_decay, set_bn_eval,
    #              with_grad_filter=False, filter_cfgs=-1, freeze_cfgs=None, use_sgd=False,
    #              momentum=0.9, anneling_steps=8008, scheduler_interval='step',
    #              lr_warmup=0, init_lr_prod=0.25):
    def __init__(self, backbone: str, backbone_args, num_classes, learning_rate, weight_decay, set_bn_eval,
                 num_of_finetune=None, with_grad_filter=False, filt_radius=None, use_sgd=False,
                 momentum=0.9, anneling_steps=8008, scheduler_interval='step',
                 lr_warmup=0, init_lr_prod=0.25):
        super(ClassificationModel, self).__init__()
    
        self.backbone = get_encoder(backbone, **backbone_args) # Nếu weights (trong backbone_args) được định nghĩa (ssl hoặc sswl) thì load weight từ online về (trong models/encoders/resnet.py hoặc mcunet.py hoặc mobilenet.py)        
        self.classifier = nn.Linear(self.backbone._out_channels[-1], num_classes)

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.set_bn_eval = set_bn_eval
        self.acc = Accuracy(num_classes=num_classes)
        # self.filter_cfgs = filter_cfgs
        # self.freeze_cfgs = freeze_cfgs
        
        #############################################################
        self.num_of_finetune = num_of_finetune
        filter_cfgs_ = get_all_conv_with_name(self) # Lấy 1 dict bao gồm tên và các lớp conv2d
        if num_of_finetune == "all":
            filter_cfgs = {"radius": filt_radius}
            filter_cfgs["cfgs"] = filter_cfgs_
            filter_cfgs["type"] = "conv"
        elif num_of_finetune > len(filter_cfgs_): # Nếu finetune toàn bộ
            logging.info("[Warning] number of finetuned layers is bigger than the total number of conv layers in the network => Finetune all the network")
            filter_cfgs = {"radius": filt_radius}
            filter_cfgs["cfgs"] = filter_cfgs_
            filter_cfgs["type"] = "conv"
        elif num_of_finetune is not None and num_of_finetune != 0 and num_of_finetune != "all": # Nếu có finetune nhưng không phải toàn bộ
            filter_cfgs_ = dict(list(filter_cfgs_.items())[-num_of_finetune:]) # Chỉ áp dụng filter vào num_of_finetune conv2d layer cuối
            # print("filter_cfgs_ :", filter_cfgs_)
            ####### Freeze các phần không được train (các layer ở phía trước filter_cfgs_)
            # print("Frozen components: ")
            for name, mod in self.named_modules():
                if name not in filter_cfgs_.keys() and name != '':
                    path_seq = name.split('.')
                    target = reduce(getattr, path_seq, self)
                    target.eval()
                    for param in target.parameters():
                        param.requires_grad = False
                    # print(name, end=' ')
                elif name in filter_cfgs_.keys(): # Duyệt đến layer sẽ được finetune => break. Vì đằng sau các conv2d layer này còn có thể có các lớp fc, gì gì đó mà vẫn cần finetune
                    break
            # print("")
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
                        param.requires_grad = False
            filter_cfgs = -1
        else:
            filter_cfgs = -1
        #######################################################################
        self.with_grad_filter = with_grad_filter
        self.use_sgd = use_sgd
        self.momentum = momentum
        self.anneling_steps = anneling_steps
        self.scheduler_interval = scheduler_interval
        self.lr_warmup = lr_warmup
        self.init_lr_prod = init_lr_prod

        self.filt_radius = filt_radius

        # self.hook = Conv2dSizeHook() # Kiểu hook cũ, 2 attribute input_size và output_size bên trong là ở dạng dict với key là module (không phải name module) và value là input/output size
        self.hook = {} # Kiểu hook mới, với hook là một dict: Trong đó key là tên module và value là hook chứa 2 attribute là input và output size của module đó.

        
        # print("Before registering filter: Weight size: ", get_total_weight_size(self))
        logging.info(f"Before registering filter: Weight size: {get_total_weight_size(self)}")
        if with_grad_filter:
            # self.radius = filter_cfgs['filter_install'][0]['radius']
            register_filter(self, filter_cfgs)
            # register_filter(self, filter_cfgs)
            # print("After registering filter: Weight size: ", get_total_weight_size(self))
            logging.info(f"After registering filter: Weight size: {get_total_weight_size(self)}")

        # Đăng kí hook
        # register_hook_for_conv(self, self.hook, consider_active_only=True, freeze_cfgs=self.freeze_cfgs) # Dùng cho hàm tính active activation size, dùng cho kiểu hook cũ (Conv2dSizeHook)
        
        # attach_hooks_for_conv(self, consider_active_only=True) # Dùng cho kiểu hook mới


        # freeze_layers(self, freeze_cfgs)
        self.acc.reset()
    def activate_hooks(self, is_activated=True):
        for h in self.hook:
            self.hook[h].activate(is_activated)

    def remove_hooks(self):
        for h in self.hook:
            self.hook[h].remove()
    def get_activation_size(self, train_data_size, consider_active_only=False, element_size=4, unit="MB"): # element_size = 4 bytes
        # Register hook to log input/output size
        attach_hooks_for_conv(self, consider_active_only=consider_active_only)
        self.activate_hooks(True)
        # Create a temporary input for model so the hook can record input/output size
        _ = th.rand(1, train_data_size[0], train_data_size[1], train_data_size[2]) # Tương đương 1 batch, train_data_size[0] kênh, train_data_size[1]xtrain_data_size[2] kích thước
        _ = self(_)
        #############################################################################
        _, first_hook = next(iter(self.hook.items()))
        if first_hook.active:
            logging.info("Hook is activated")
        else:
            logging.info("[Warning] Hook is not activated !!")
        if not consider_active_only:
            conv2d_layers = get_all_conv_with_name(self)
        else:
            conv2d_layers = get_active_conv_with_name(self)
        assert conv2d_layers != -1, "[Warning] Consider activate conv2d only but no conv2d is finetuned => No hook is attached !!"

        num_element = 0
        for name in self.hook:
            input_size = th.tensor(self.hook[name].input_size).clone().detach()
            stride = conv2d_layers[name].stride
            x_h, x_w = input_size[-2:]
            h, w = self.hook[name].output_size[-2:]

            if isinstance(self.hook[name].module, Conv2dAvg): # Nếu key là Conv2dAVG    
                p_h, p_w = ceil(h / self.filt_radius), ceil(w / self.filt_radius)
                x_order_h, x_order_w = self.filt_radius * stride[0], self.filt_radius * stride[1]
                x_pad_h, x_pad_w = ceil((p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)

                x_sum_height = ((x_h + 2 * x_pad_h - x_order_h) // x_order_h) + 1
                x_sum_width = ((x_w + 2 * x_pad_w - x_order_w) // x_order_w) + 1

                num_element += int(1 * input_size[1] * x_sum_height * x_sum_width) # Bỏ qua số batch

            elif isinstance(self.hook[name].module, Conv2dSVD): # SVD case
                from custom_op.conv_svd import truncated_svd
                temp_tensor = th.rand(1, input_size[1], input_size[2], input_size[3], dtype=th.float32)
                U, S, Vt, Uk, Sk, Vk_t = truncated_svd(temp_tensor, k=1)
                num_element += int(Uk.shape[1]*Uk.shape[2]*Uk.shape[3] + Sk.shape[1]*Sk.shape[2] + Vk_t.shape[1]*Vk_t.shape[2]*Vk_t.shape[3])
                temp_tensor = None # giải phóng bộ nhớ
            
            elif isinstance(self.hook[name].module, nn.modules.conv.Conv2d): # Nếu key là Conv2d
                num_element += int(1 * input_size[1] * input_size[2] * input_size[3]) # Bỏ qua số batch, Lưu luôn như này do trong hàm forward của convavg, nó lưu thẳng x sau forward
        self.remove_hooks()

        if unit == "MB":
            return str(round(num_element*element_size/(1024*1024), 2)) + " MB"
        elif unit == "KB":
            return str(round(num_element*element_size/(1024), 2)) + " KB"
        else:
            raise ValueError("Unit is not suitable")
    # def get_activation_size(self, train_data_size, consider_active_only=False, element_size=4, unit="MB"): # element_size = 4 bytes
    #     # Register hook to log input/output size
    #     attach_hooks_for_conv(self, consider_active_only=consider_active_only)
    #     self.activate_hooks(True)
    #     # Create a temporary input for model so the hook can record input/output size
    #     _ = th.rand(1, train_data_size[0], train_data_size[1], train_data_size[2]) # Tương đương 1 batch, train_data_size[0] kênh, train_data_size[1]xtrain_data_size[2] kích thước
    #     _ = self(_)
    #     #############################################################################
    #     _, first_hook = next(iter(self.hook.items()))
    #     if first_hook.active:
    #         print("Hook is activated")
    #     else:
    #         print("[Warning] Hook is not activated !!")
    #     if not consider_active_only:
    #         conv2d_layers = get_all_conv_with_name(self)
    #     else:
    #         if self.num_of_finetune == "all":
    #             conv2d_layers = get_all_conv_with_name(self)
    #         else:
    #             conv2d_layers = get_active_conv_with_name(self, self.freeze_cfgs)

    #     num_element = 0

    #     for name in self.hook:
    #         input_size = th.tensor(self.hook[name].input_size).clone().detach()
    #         stride = conv2d_layers[name].stride
    #         x_h, x_w = input_size[-2:]
    #         h, w = self.hook[name].output_size[-2:]

    #         if isinstance(self.hook[name].module, Conv2dAvg): # Nếu key là Conv2dAVG    
    #             p_h, p_w = ceil(h / self.radius), ceil(w / self.radius)
    #             x_order_h, x_order_w = self.radius * stride[0], self.radius * stride[1]
    #             x_pad_h, x_pad_w = ceil((p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)

    #             x_sum_height = ((x_h + 2 * x_pad_h - x_order_h) // x_order_h) + 1
    #             x_sum_width = ((x_w + 2 * x_pad_w - x_order_w) // x_order_w) + 1

    #             num_element += int(1 * input_size[1] * x_sum_height * x_sum_width) # Bỏ qua số batch

    #         elif isinstance(self.hook[name].module, Conv2dSVD): # SVD case
    #             from custom_op.conv_svd import truncated_svd
    #             temp_tensor = th.rand(1, input_size[1], input_size[2], input_size[3], dtype=th.float32)
    #             U, S, Vt, Uk, Sk, Vk_t = truncated_svd(temp_tensor, k=1)
    #             num_element += int(Uk.shape[1]*Uk.shape[2]*Uk.shape[3] + Sk.shape[1]*Sk.shape[2] + Vk_t.shape[1]*Vk_t.shape[2]*Vk_t.shape[3])
    #             temp_tensor = None # giải phóng bộ nhớ
            
    #         elif isinstance(self.hook[name].module, nn.modules.conv.Conv2d): # Nếu key là Conv2d
    #             num_element += int(1 * input_size[1] * input_size[2] * input_size[3]) # Bỏ qua số batch, Lưu luôn như này do trong hàm forward của convavg, nó lưu thẳng x sau forward
    #     self.remove_hooks()

    #     if unit == "MB":
    #         return str(round(num_element*element_size/(1024*1024), 2)) + " MB"
    #     elif unit == "KB":
    #         return str(round(num_element*element_size/(1024), 2)) + " KB"
    #     else:
    #         raise ValueError("Unit is not suitable")
    
    # def get_activation_size_old(self, consider_active_only=False, element_size=4, unit="MB"): # element_size = 4 bytes # Dùng cho kiểu hook cũ
    #     if not consider_active_only:
    #         conv2d_layers = get_all_conv(self)
    #     else:
    #         if self.freeze_cfgs == None:
    #             conv2d_layers = get_all_conv(self)
    #         else:
    #             conv2d_layers = get_active_conv(self, self.freeze_cfgs)

    #     input_sizes = self.hook.input_size
    #     output_sizes = self.hook.output_size
    #     num_element = 0
    #     idx = 0
    #     for key, input_size in input_sizes.items():
    #         input_size = th.tensor(input_size)
    #         stride = conv2d_layers[idx].stride
    #         x_h, x_w = input_size[-2:]
    #         h, w = output_sizes[key][-2:]

    #         if isinstance(key, Conv2dAvg): # Nếu key là Conv2dAVG    
    #             p_h, p_w = ceil(h / self.radius), ceil(w / self.radius)
    #             x_order_h, x_order_w = self.radius * stride[0], self.radius * stride[1]
    #             x_pad_h, x_pad_w = ceil((p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)

    #             x_sum_height = ((x_h + 2 * x_pad_h - x_order_h) // x_order_h) + 1
    #             x_sum_width = ((x_w + 2 * x_pad_w - x_order_w) // x_order_w) + 1

    #             num_element += int(1 * input_size[1] * x_sum_height * x_sum_width) # Bỏ qua số batch

    #         elif isinstance(key, Conv2dSVD): ######################################
    #             from custom_op.conv_svd import truncated_svd
    #             temp_tensor = th.rand(1, input_size[1], input_size[2], input_size[3], dtype=th.float32)
    #             U, S, Vt, Uk, Sk, Vk_t = truncated_svd(temp_tensor, k=1)
    #             num_element += int(Uk.shape[1]*Uk.shape[2]*Uk.shape[3] + Sk.shape[1]*Sk.shape[2] + Vk_t.shape[1]*Vk_t.shape[2]*Vk_t.shape[3])
    #             temp_tensor = None # giải phóng bộ nhớ
    #             # num_element += int(1 * input_size[1] * input_size[2] * input_size[3]) # Bỏ qua số batch, Lưu luôn như này do trong hàm forward của convavg, nó lưu thẳng x sau forward


    #         elif isinstance(key, nn.modules.conv.Conv2d): # Nếu key là Conv2d
    #             num_element += int(1 * input_size[1] * input_size[2] * input_size[3]) # Bỏ qua số batch, Lưu luôn như này do trong hàm forward của convavg, nó lưu thẳng x sau forward
    #         idx += 1
    #     if unit == "MB":
    #         return str(round(num_element*element_size/(1024*1024), 2)) + " MB"
    #     elif unit == "KB":
    #         return str(round(num_element*element_size/(1024), 2)) + " KB"
    #     else:
    #         raise ValueError("Unit is not suitable")
    
    def register_filter(self):
        register_filter(self, self.filter_cfgs)

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
        feat = self.backbone(x)[-1]
        feat = self.pooling(feat)
        feat = feat.flatten(start_dim=1)
        logit = self.classifier(feat)
        return logit

    def training_step(self, train_batch, batch_idx):
        if self.set_bn_eval:
            self.bn_eval()
        img, label = train_batch['image'], train_batch['label'] # Lấy dữ liệu
        if img.shape[1] == 1:
            img = th.cat([img] * 3, dim=1)
        logits = self.forward(img) # Lấy output predict
        pred_cls = th.argmax(logits, dim=-1)
        acc = th.sum(pred_cls == label) / label.shape[0]
        loss = self.loss(logits, label) # Tính loss giữa output predict và true output
        self.log("Train/Loss", loss)
        self.log("Train/Acc", acc)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        with open(os.path.join(self.logger.log_dir, 'train_loss.log'), 'a') as f:
            mean = th.stack([o['loss'] for o in outputs]).mean()
            f.write(f"{self.current_epoch} {mean}")
            f.write("\n")

    def validation_step(self, val_batch, batch_idx):
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
        f = open(os.path.join(self.logger.log_dir, 'val.log'),
                 'a') if self.logger is not None else None
        acc = self.acc.compute()
        if self.logger is not None:
            f.write(f"{self.current_epoch} {acc}\n")
            f.close()
        self.log("Val/Acc", acc)
        self.log("val-acc", acc)
        self.acc.reset()
