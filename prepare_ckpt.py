# This function is used to create backbone model for the code (It is not important)
import torch
import os
from torchvision.models import (
    resnet34,
    resnet18,
    resnet50,
    mobilenet_v2,
    swin_t,
    ResNet18_Weights,
    ResNet50_Weights,
    MobileNet_V2_Weights,
    ResNet34_Weights,
    Swin_T_Weights
)

from pytorch_lightning import LightningModule
from models.encoders import get_encoder


# Thêm attribute backbone vào mỗi pretrained model
class ClassificationModel(LightningModule):
    def __init__(self, backbone: str):
        super(ClassificationModel, self).__init__()
        if backbone == "resnet18":
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone == "resnet34":
            self.backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif backbone == "resnet50":
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif backbone == "mbv2":
            self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        elif backbone == "swinT":
            self.backbone = swin_t(weights=Swin_T_Weights.DEFAULT)
        else:
            raise ValueError(f"No such model {backbone}")

# Hàm để lưu chúng
def save_full_ckpt(net_name, saved_location, turn_to_backbone=True):
    if os.path.exists(saved_location) == False:
        os.mkdir(saved_location)
    if turn_to_backbone:
        name = net_name + ".ckpt"
        if net_name == "pre_trained_resnet18":
            model = ClassificationModel(backbone='resnet18')
        elif net_name == "pre_trained_resnet34":
            model = ClassificationModel(backbone='resnet34')
        elif net_name == "pre_trained_resnet50":
            model = ClassificationModel(backbone='resnet50')
        elif net_name == "pre_trained_mbv2":
            model = ClassificationModel(backbone='mbv2')
        elif net_name == "pre_trained_swinT":
            model = ClassificationModel(backbone='swinT')
        else:
            raise ValueError(f"No such model {net_name}")
    else:
        name = net_name + "_raw.ckpt"
        if net_name == "pre_trained_resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif net_name == "pre_trained_resnet34":
            model = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif net_name == "pre_trained_resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif net_name == "pre_trained_mbv2":
            model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        elif net_name == "pre_trained_swinT":
            model = swin_t(weights=Swin_T_Weights.DEFAULT)
        elif net_name == "resnet18":
            model = resnet18()
        elif net_name == "resnet34":
            model = resnet34()
        elif net_name == "resnet50":
            model = resnet50()
        elif net_name == "mbv2":
            model = mobilenet_v2()
        elif net_name == "swinT":
            model = swin_t()
        else:
            raise ValueError(f"No such model {net_name}")

    
    checkpoint_path = os.path.join(saved_location, name)
    
    # Lưu checkpoint
    torch.save({
        'state_dict': model.state_dict(),
    }, checkpoint_path)

    print("Checkpoint đã được lưu tại:", checkpoint_path)



# names = ["pre_trained_resnet18", "pre_trained_resnet34", "pre_trained_resnet50", "pre_trained_mbv2"]
# names = ["resnet18", "resnet50", "mbv2"]

saved_location = "./pretrained_ckpts/"
# for name in names:
    # save_full_ckpt(name, saved_location, False)

# save_full_ckpt("swinT", saved_location, False)



# def load_model(name, saved_location):
#     # saved_location = os.path.join(saved_location, name + ".ckpt")
#     # model = resnet18()
#     # backbone = 'resnet18'
#     # backbone_args = {
#     #     'in_channels': 3,
#     #     'output_stride': 32,
#     #     'weights': "ssl"
#     # }
#     # model = get_encoder(backbone, **backbone_args) # Nếu weights (trong backbone_args) được định nghĩa (ssl hoặc sswl) thì load weight từ online về (trong models/encoders/resnet.py hoặc mcunet.py hoặc mobilenet.py)
#     model_state_dict = torch.load(saved_location)#['state_dict']
#     # model.load_state_dict(model_state_dict)
#     print(model_state_dict['state_dict'])
#     # print(model.backbone)
#     # print(model.keys())

# # load_model("pre_trained_resnet18",saved_location)
# load_model("swinT","/home/infres/lnguyen-23/test/classification/pretrained_ckpts/swinT_raw.ckpt")



