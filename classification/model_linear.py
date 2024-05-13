# Bản của mình
import os
import numpy as np
import torch as th
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from custom_op_linear.register import register_HOSVD_with_var, register_avg_batch
from custom_op_linear.conv_avg_batch import Conv2d_Avg_Batch
from custom_op_linear.linear_hosvd_with_var import Linear_HOSVD
from custom_op_linear.linear_avg_batch import Linear_avg_batch
from util import freeze_layers, get_total_weight_size, get_all_linear_with_name, attach_hooks_for_linear
from math import ceil
from models.encoders import get_encoder
from functools import reduce
import logging

class ClassificationModel(LightningModule):
    def __init__(self, backbone: str, backbone_args, num_classes,
                 learning_rate, weight_decay, set_bn_eval, load = None,

                 num_of_finetune=None, 
                 with_avg_batch=False, with_HOSVD_with_var_compression = False,
                 SVD_var=None,
                 
                 use_sgd=False, momentum=0.9, anneling_steps=8008, scheduler_interval='step',
                 lr_warmup=0, init_lr_prod=0.25):
        super(ClassificationModel, self).__init__()
        self.backbone_name = backbone
        self.backbone = get_encoder(backbone, **backbone_args) # Nếu weights (trong backbone_args) được định nghĩa (ssl hoặc sswl) thì load weight từ online về (trong models/encoders/resnet.py hoặc mcunet.py hoặc mobilenet.py)        
        self.backbone.head = nn.Linear(in_features=768, out_features=num_classes, bias=True) # Thay lớp cuối của swinT
            
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.set_bn_eval = set_bn_eval
        self.acc = Accuracy(num_classes=num_classes)
        ##
        self.with_avg_batch = with_avg_batch
        self.with_HOSVD_with_var_compression = with_HOSVD_with_var_compression
        self.SVD_var = SVD_var
        self.use_sgd = use_sgd
        self.momentum = momentum
        self.anneling_steps = anneling_steps
        self.scheduler_interval = scheduler_interval
        self.lr_warmup = lr_warmup
        self.init_lr_prod = init_lr_prod
        self.hook = {} # Hook being a dict: where key is the module name and value is the hook
        self.num_of_finetune = num_of_finetune
        #############################################################
        filter_cfgs_ = get_all_linear_with_name(self) # Lấy 1 dict bao gồm tên và các lớp linear
        filter_cfgs = {}

        if num_of_finetune == "all":
            if with_HOSVD_with_var_compression:
                filter_cfgs = {"SVD_var": SVD_var}

            filter_cfgs["cfgs"] = filter_cfgs_
        elif num_of_finetune > len(filter_cfgs_): # Nếu finetune toàn bộ
            logging.info("[Warning] number of finetuned layers is bigger than the total number of linear layers in the network => Finetune all the network")
            if with_HOSVD_with_var_compression:
                filter_cfgs = {"SVD_var": SVD_var}

            filter_cfgs["cfgs"] = filter_cfgs_
        elif num_of_finetune is not None and num_of_finetune != 0 and num_of_finetune != "all": # Nếu có finetune nhưng không phải toàn bộ
            filter_cfgs_ = dict(list(filter_cfgs_.items())[-num_of_finetune:]) # Chỉ áp dụng filter vào num_of_finetune linear layer cuối
            for name, mod in self.named_modules():
                if len(list(mod.children())) == 0 and name not in filter_cfgs_.keys() and name != '':
                    mod.eval()
                    for param in mod.parameters():
                        param.requires_grad = False # Freeze layer
                elif name in filter_cfgs_.keys(): # Khi duyệt đến layer sẽ được finetune => break. Vì đằng sau các linear layer này còn có thể có các lớp fc, gì gì đó mà vẫn cần finetune
                    break
            if with_HOSVD_with_var_compression:
                filter_cfgs = {"SVD_var": SVD_var}

            filter_cfgs["cfgs"] = filter_cfgs_
        
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
            register_avg_batch(self, filter_cfgs)



        self.acc.reset()
    def activate_hooks(self, is_activated=True):
        for h in self.hook:
            self.hook[h].activate(is_activated)

    def remove_hooks(self):
        for h in self.hook:
            self.hook[h].remove()
        logging.info("Hook is removed")

    def get_activation_size(self, trainer, data, consider_active_only=False, element_size=4, unit="MB"): # element_size = 4 bytes
        # Register hook to log input/output size
        train_data_size = th.tensor([data.batch_size, 3, data.width, data.height]) # 3 channels
        attach_hooks_for_linear(self, consider_active_only=consider_active_only)
        self.activate_hooks(True)
        #############################################################################
        _, first_hook = next(iter(self.hook.items()))
        if first_hook.active:
            logging.info("Hook is activated")
        else:
            logging.info("[Warning] Hook is not activated !!")
        #############################################################################
        # Create a temporary input for model so the hook can record input/output size
        # trainer.validate(self, datamodule=data)  

        if isinstance(first_hook.module, Linear_HOSVD):
            trainer.validate(self, datamodule=data)  
        elif isinstance(first_hook.module, nn.modules.linear.Linear):
            _ = th.rand(train_data_size[0], train_data_size[1], train_data_size[2], train_data_size[3])
            _ = self(_)

        num_element = 0
        for name in self.hook:
            # print("name: ", name)
            # print("module: ", self.hook[name].module)
            input_size = th.tensor(self.hook[name].input_size).clone().detach()
            # x_h, x_w = input_size[-2:]
            # h, w = self.hook[name].output_size[-2:]

            # if isinstance(self.hook[name].module, Conv2dAvg): 
            #     p_h, p_w = ceil(h / self.filt_radius), ceil(w / self.filt_radius)
            #     x_order_h, x_order_w = self.filt_radius * stride[0], self.filt_radius * stride[1]
            #     x_pad_h, x_pad_w = ceil((p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)

            #     x_sum_height = ((x_h + 2 * x_pad_h - x_order_h) // x_order_h) + 1
            #     x_sum_width = ((x_w + 2 * x_pad_w - x_order_w) // x_order_w) + 1

            #     num_element += int(input_size[0] * input_size[1] * x_sum_height * x_sum_width)

                # print(name, ": ", self.hook[name].module, " ----- ",int(input_size[0] * input_size[1] * x_sum_height * x_sum_width))

            if isinstance(self.hook[name].module, Conv2d_Avg_Batch):
                temp_tensor = th.rand(input_size[0], input_size[1], input_size[2], input_size[3], dtype=th.float32)  
                from custom_op.conv_avg_batch import conv_avg_batch
                x = conv_avg_batch(temp_tensor)
                num_element += x.numel() + th.tensor(temp_tensor.shape).numel()

            
            elif isinstance(self.hook[name].module, Linear_HOSVD):
                print(name, " ", self.hook[name].module, " ", len(self.hook[name].inputs))
                ############## Kiểu reshape activation map theo dim
                from custom_op_linear.linear_hosvd_with_var import hosvd
                num_element_all = 0
                for input in self.hook[name].inputs:
                    if input.dim() == 2:
                        S, u0, u1 = hosvd(input, var=self.SVD_var)
                        num_element_all += S.numel() + u0.numel() + u1.numel()
                    else:
                        S, u0, u1, u2, u3 = hosvd(input, var=self.SVD_var)
                        num_element_all += S.numel() + u0.numel() + u1.numel() + u2.numel() + u3.numel()
                if len(self.hook[name].inputs) != 0:
                    num_element += num_element_all/len(self.hook[name].inputs)
            
            elif isinstance(self.hook[name].module, nn.modules.linear.Linear):
                print(name, " ", self.hook[name].module, " ", len(self.hook[name].inputs), " ", int(input_size.prod()))
                num_element += int(input_size.prod())

        self.remove_hooks()

        if unit == "Byte":
            return str(num_element*element_size) + " Bytes"
        if unit == "MB":
            return str(round((num_element*element_size)/(1024*1024), 2)) + " MB"
        elif unit == "KB":
            return str(round((num_element*element_size)/(1024), 2)) + " KB"
            # return num_element
        else:
            raise ValueError("Unit is not suitable")


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
        logit = self.backbone(x)
        return logit

    def training_step(self, train_batch, batch_idx):
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
        with open(os.path.join(self.logger.log_dir, 'train_loss.log'), 'a') as f:
            mean_loss = th.stack([o['loss'] for o in outputs]).mean()
            f.write(f"{self.current_epoch} {mean_loss}")
            f.write("\n")

        with open(os.path.join(self.logger.log_dir, 'train_acc.log'), 'a') as f:
            mean_acc = th.stack([o['acc'] for o in outputs]).mean()
            f.write(f"{self.current_epoch} {mean_acc}")
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