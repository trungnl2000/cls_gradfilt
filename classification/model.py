# Bản của mình
import os
import numpy as np
import torch as th
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from custom_op.register import register_filter
from custom_op.conv_avg import Conv2dAvg
from util import freeze_layers, get_total_weight_size, Conv2dSizeHook, register_hook_for_conv, get_all_conv, get_active_conv
from math import ceil
from models.encoders import get_encoder

class ClassificationModel(LightningModule):
    def __init__(self, backbone: str, backbone_args, num_classes, learning_rate, weight_decay, set_bn_eval,
                 with_grad_filter=False, filter_cfgs=-1, freeze_cfgs=None, use_sgd=False,
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
        self.filter_cfgs = filter_cfgs
        self.freeze_cfgs = freeze_cfgs
        self.with_grad_filter = with_grad_filter
        self.use_sgd = use_sgd
        self.momentum = momentum
        self.anneling_steps = anneling_steps
        self.scheduler_interval = scheduler_interval
        self.lr_warmup = lr_warmup
        self.init_lr_prod = init_lr_prod

        self.radius = None
        self.hook = Conv2dSizeHook()
        
        print("Before registering filter: Weight size: ", get_total_weight_size(self))

        if with_grad_filter:
            self.radius = filter_cfgs['filter_install'][0]['radius']
            register_filter(self, filter_cfgs)
            print("After registering filter: Weight size: ", get_total_weight_size(self))

        register_hook_for_conv(self, self.hook, consider_active_only=True, freeze_cfgs=self.freeze_cfgs)


        freeze_layers(self, freeze_cfgs)
        self.acc.reset()
        
    def get_activation_size(self, consider_active_only=False, element_size=4, unit="MB"): # element_size = 4 bytes
        if not consider_active_only:
            conv2d_layers = get_all_conv(self)
        else:
            if self.freeze_cfgs == None:
                conv2d_layers = get_all_conv(self)
            else:
                conv2d_layers = get_active_conv(self, self.freeze_cfgs)

        input_sizes = self.hook.input_size
        output_sizes = self.hook.output_size
        num_element = 0
        idx = 0
        for key, input_size in input_sizes.items():
            input_size = th.tensor(input_size)
            stride = conv2d_layers[idx].stride
            x_h, x_w = input_size[-2:]
            h, w = output_sizes[key][-2:]

            if isinstance(key, Conv2dAvg): # Nếu key là Conv2dAVG    
                p_h, p_w = ceil(h / self.radius), ceil(w / self.radius)
                x_order_h, x_order_w = self.radius * stride[0], self.radius * stride[1]
                x_pad_h, x_pad_w = ceil((p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)

                x_sum_height = ((x_h + 2 * x_pad_h - x_order_h) // x_order_h) + 1
                x_sum_width = ((x_w + 2 * x_pad_w - x_order_w) // x_order_w) + 1

                num_element += int(1 * input_size[1] * x_sum_height * x_sum_width) # Bỏ qua số batch

            elif isinstance(key, nn.modules.conv.Conv2d): # Nếu key là Conv2d
                # padding_size = key.padding
                # kernel_size = key.kernel_size
                # x_height = ((x_h + 2 * padding_size[0] - kernel_size[0]) // stride[0]) + 1
                # x_width = ((x_w + 2 * padding_size[1] - kernel_size[1]) // stride[1]) + 1
                # num_element += int(1 * input_size[1] * x_height * x_width) # Bỏ qua số batch
                num_element += int(1 * input_size[1] * input_size[2] * input_size[3]) # Bỏ qua số batch, Lưu luôn như này do trong hàm forward của convavg, nó lưu thẳng x sau forward
            idx += 1
        if unit == "MB":
            return str(round(num_element*element_size/(1024*1024), 2)) + " MB"
        elif unit == "KB":
            return str(round(num_element*element_size/(1024), 2)) + " KB"
        else:
            raise ValueError("Unit is not suitable")
    
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
