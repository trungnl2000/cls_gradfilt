import logging
from copy import deepcopy
from functools import reduce
import torch.nn as nn
from .conv_avg import wrap_conv_layer
from .conv_svd import wrap_convSVD_layer


DEFAULT_CFG = {
    "path": "",
    "radius": 8,
    "type": "",
}


def add_grad_filter(module: nn.Module, cfg):
    if cfg['type'] == 'cbr':
        module.conv = wrap_conv_layer(module.conv, cfg['radius'], True)
    elif cfg['type'] == 'resnet_basic_block':
        module.conv1 = wrap_conv_layer(module.conv1, cfg['radius'], True)
        module.conv2 = wrap_conv_layer(module.conv2, cfg['radius'], True)
    elif cfg['type'] == 'resnet_bottleneck_block':
        module.conv1 = wrap_conv_layer(module.conv1, cfg['radius'], True)
        module.conv2 = wrap_conv_layer(module.conv2, cfg['radius'], True)
        module.conv3 = wrap_conv_layer(module.conv3, cfg['radius'], True)
    elif cfg['type'] == 'resnet_layer':
        assert isinstance(module, nn.Sequential)
        ccfg = deepcopy(cfg)
        ccfg['type'] = 'resnet_basic_block'
        for blk_idx, blk in enumerate(module):
            add_grad_filter(blk, ccfg)
    elif cfg['type'] == 'ConvNormActivation':
        module[0] = wrap_conv_layer(module[0], cfg['radius'], True)
    elif cfg['type'] == 'InvertedResidual':
        count = len(module.conv)
        module.conv[-2] = wrap_conv_layer(module.conv[-2], cfg['radius'], True)
        for i in range(count - 2):
            module.conv[i][0] = wrap_conv_layer(module.conv[i][0], cfg['radius'], True)
    elif cfg['type'] == "MCUNetBlock":
        # if hasattr(module.mobile_inverted_conv, 'inverted_bottleneck'):
        if module.mobile_inverted_conv.inverted_bottleneck is not None:
            module.mobile_inverted_conv.inverted_bottleneck.conv = wrap_conv_layer(
                module.mobile_inverted_conv.inverted_bottleneck.conv, cfg['radius'], True)
        # if hasattr(module.mobile_inverted_conv, 'depth_conv'):
        if module.mobile_inverted_conv.depth_conv is not None:
            module.mobile_inverted_conv.depth_conv.conv = wrap_conv_layer(
                module.mobile_inverted_conv.depth_conv.conv, cfg['radius'], True)
        # if hasattr(module.mobile_inverted_conv, 'point_linear'):
        if module.mobile_inverted_conv.point_linear is not None:
            module.mobile_inverted_conv.point_linear.conv = wrap_conv_layer(
                module.mobile_inverted_conv.point_linear.conv, cfg['radius'], True)
    elif cfg['type'] == 'conv':
        module = wrap_conv_layer(module, cfg['radius'], True)
    else:
        raise NotImplementedError
    return module


def add_svd_filter(module: nn.Module, cfg):
    if cfg['type'] == 'cbr':
        module.conv = wrap_convSVD_layer(module.conv, cfg["truncated_SVD_k"], True)
    elif cfg['type'] == 'resnet_basic_block':
        module.conv1 = wrap_convSVD_layer(module.conv1, cfg["truncated_SVD_k"], True)
        module.conv2 = wrap_convSVD_layer(module.conv2, cfg["truncated_SVD_k"], True)
    elif cfg['type'] == 'resnet_bottleneck_block':
        module.conv1 = wrap_convSVD_layer(module.conv1, cfg["truncated_SVD_k"], True)
        module.conv2 = wrap_convSVD_layer(module.conv2, cfg["truncated_SVD_k"], True)
        module.conv3 = wrap_convSVD_layer(module.conv3, cfg["truncated_SVD_k"], True)
    elif cfg['type'] == 'resnet_layer':
        assert isinstance(module, nn.Sequential)
        ccfg = deepcopy(cfg)
        ccfg['type'] = 'resnet_basic_block'
        for blk_idx, blk in enumerate(module):
            add_svd_filter(blk, ccfg)
    elif cfg['type'] == 'ConvNormActivation':
        module[0] = wrap_convSVD_layer(module[0], cfg["truncated_SVD_k"], True)
    elif cfg['type'] == 'InvertedResidual':
        count = len(module.conv)
        module.conv[-2] = wrap_convSVD_layer(module.conv[-2], cfg["truncated_SVD_k"], True)
        for i in range(count - 2):
            module.conv[i][0] = wrap_convSVD_layer(module.conv[i][0], cfg["truncated_SVD_k"], True)
    elif cfg['type'] == "MCUNetBlock":
        # if hasattr(module.mobile_inverted_conv, 'inverted_bottleneck'):
        if module.mobile_inverted_conv.inverted_bottleneck is not None:
            module.mobile_inverted_conv.inverted_bottleneck.conv = wrap_convSVD_layer(
                module.mobile_inverted_conv.inverted_bottleneck.conv, cfg["truncated_SVD_k"], True)
        # if hasattr(module.mobile_inverted_conv, 'depth_conv'):
        if module.mobile_inverted_conv.depth_conv is not None:
            module.mobile_inverted_conv.depth_conv.conv = wrap_convSVD_layer(
                module.mobile_inverted_conv.depth_conv.conv, cfg["truncated_SVD_k"], True)
        # if hasattr(module.mobile_inverted_conv, 'point_linear'):
        if module.mobile_inverted_conv.point_linear is not None:
            module.mobile_inverted_conv.point_linear.conv = wrap_convSVD_layer(
                module.mobile_inverted_conv.point_linear.conv, cfg["truncated_SVD_k"], True)
    elif cfg['type'] == 'conv':
        module = wrap_convSVD_layer(module, cfg["truncated_SVD_k"], True)
    else:
        raise NotImplementedError
    return module



def register_filter(module, cfgs):
    # filter_install_cfgs = cfgs['filter_install']
    logging.info("Registering Filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["cfgs"]: # Dò tên của từng conv layer sẽ bị cài filter
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_grad_filter(target, cfgs)

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_SVD(module, cfgs):
    # filter_install_cfgs = cfgs['filter_install']
    logging.info("Registering Filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["cfgs"]: # Dò tên của từng conv layer sẽ bị cài filter
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_svd_filter(target, cfgs)

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)