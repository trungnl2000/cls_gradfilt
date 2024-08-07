import logging
from copy import deepcopy
from functools import reduce
import torch.nn as nn
from .conv_avg import wrap_conv_layer
from .conv_svd_with_var import wrap_convSVD_with_var_layer
from .conv_avg_batch import wrap_conv_Avg_Batch_layer
from .conv_hosvd_with_var import wrap_convHOSVD_with_var_layer
from .conv_gkpd_hosvd import wrap_conv_GKPD_HOSVD_layer


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

def add_svd_with_var_filter(module: nn.Module, cfg):
    if cfg['type'] == 'cbr':
        module.conv = wrap_convSVD_with_var_layer(module.conv, cfg["SVD_var"], True, cfg["svd_size"])
    elif cfg['type'] == 'resnet_basic_block':
        module.conv1 = wrap_convSVD_with_var_layer(module.conv1, cfg["SVD_var"], True, cfg["svd_size"])
        module.conv2 = wrap_convSVD_with_var_layer(module.conv2, cfg["SVD_var"], True, cfg["svd_size"])
    elif cfg['type'] == 'resnet_bottleneck_block':
        module.conv1 = wrap_convSVD_with_var_layer(module.conv1, cfg["SVD_var"], True, cfg["svd_size"])
        module.conv2 = wrap_convSVD_with_var_layer(module.conv2, cfg["SVD_var"], True, cfg["svd_size"])
        module.conv3 = wrap_convSVD_with_var_layer(module.conv3, cfg["SVD_var"], True, cfg["svd_size"])
    elif cfg['type'] == 'resnet_layer':
        assert isinstance(module, nn.Sequential)
        ccfg = deepcopy(cfg)
        ccfg['type'] = 'resnet_basic_block'
        for blk_idx, blk in enumerate(module):
            add_svd_with_var_filter(blk, ccfg)
    elif cfg['type'] == 'ConvNormActivation':
        module[0] = wrap_convSVD_with_var_layer(module[0], cfg["SVD_var"], True, cfg["svd_size"])
    elif cfg['type'] == 'InvertedResidual':
        count = len(module.conv)
        module.conv[-2] = wrap_convSVD_with_var_layer(module.conv[-2], cfg["SVD_var"], True, cfg["svd_size"])
        for i in range(count - 2):
            module.conv[i][0] = wrap_convSVD_with_var_layer(module.conv[i][0], cfg["SVD_var"], True, cfg["svd_size"])
    elif cfg['type'] == "MCUNetBlock":
        # if hasattr(module.mobile_inverted_conv, 'inverted_bottleneck'):
        if module.mobile_inverted_conv.inverted_bottleneck is not None:
            module.mobile_inverted_conv.inverted_bottleneck.conv = wrap_convSVD_with_var_layer(
                module.mobile_inverted_conv.inverted_bottleneck.conv, cfg["SVD_var"], True, cfg["svd_size"])
        # if hasattr(module.mobile_inverted_conv, 'depth_conv'):
        if module.mobile_inverted_conv.depth_conv is not None:
            module.mobile_inverted_conv.depth_conv.conv = wrap_convSVD_with_var_layer(
                module.mobile_inverted_conv.depth_conv.conv, cfg["SVD_var"], True, cfg["svd_size"])
        # if hasattr(module.mobile_inverted_conv, 'point_linear'):
        if module.mobile_inverted_conv.point_linear is not None:
            module.mobile_inverted_conv.point_linear.conv = wrap_convSVD_with_var_layer(
                module.mobile_inverted_conv.point_linear.conv, cfg["SVD_var"], True, cfg["svd_size"])
    elif cfg['type'] == 'conv':
        module = wrap_convSVD_with_var_layer(module, cfg["SVD_var"], True, cfg["svd_size"])
    else:
        raise NotImplementedError
    return module

def add_avg_batch_layer(module: nn.Module, cfg):
    if cfg['type'] == 'cbr':
        module.conv = wrap_conv_Avg_Batch_layer(module.conv, True)
    elif cfg['type'] == 'resnet_basic_block':
        module.conv1 = wrap_conv_Avg_Batch_layer(module.conv1, True)
        module.conv2 = wrap_conv_Avg_Batch_layer(module.conv2, True)
    elif cfg['type'] == 'resnet_bottleneck_block':
        module.conv1 = wrap_conv_Avg_Batch_layer(module.conv1, True)
        module.conv2 = wrap_conv_Avg_Batch_layer(module.conv2, True)
        module.conv3 = wrap_conv_Avg_Batch_layer(module.conv3, True)
    elif cfg['type'] == 'resnet_layer':
        assert isinstance(module, nn.Sequential)
        ccfg = deepcopy(cfg)
        ccfg['type'] = 'resnet_basic_block'
        for blk_idx, blk in enumerate(module):
            add_avg_batch_layer(blk, ccfg)
    elif cfg['type'] == 'ConvNormActivation':
        module[0] = wrap_conv_Avg_Batch_layer(module[0], True)
    elif cfg['type'] == 'InvertedResidual':
        count = len(module.conv)
        module.conv[-2] = wrap_conv_Avg_Batch_layer(module.conv[-2], True)
        for i in range(count - 2):
            module.conv[i][0] = wrap_conv_Avg_Batch_layer(module.conv[i][0], True)
    elif cfg['type'] == "MCUNetBlock":
        # if hasattr(module.mobile_inverted_conv, 'inverted_bottleneck'):
        if module.mobile_inverted_conv.inverted_bottleneck is not None:
            module.mobile_inverted_conv.inverted_bottleneck.conv = wrap_conv_Avg_Batch_layer(
                module.mobile_inverted_conv.inverted_bottleneck.conv, True)
        # if hasattr(module.mobile_inverted_conv, 'depth_conv'):
        if module.mobile_inverted_conv.depth_conv is not None:
            module.mobile_inverted_conv.depth_conv.conv = wrap_conv_Avg_Batch_layer(
                module.mobile_inverted_conv.depth_conv.conv, True)
        # if hasattr(module.mobile_inverted_conv, 'point_linear'):
        if module.mobile_inverted_conv.point_linear is not None:
            module.mobile_inverted_conv.point_linear.conv = wrap_conv_Avg_Batch_layer(
                module.mobile_inverted_conv.point_linear.conv, True)
    elif cfg['type'] == 'conv':
        module = wrap_conv_Avg_Batch_layer(module, True)
    else:
        raise NotImplementedError
    return module


def add_hosvd_with_var_filter(module: nn.Module, cfg):
    if cfg['type'] == 'cbr':
        module.conv = wrap_convHOSVD_with_var_layer(module.conv, cfg["SVD_var"], True, cfg["k_hosvd"])
    elif cfg['type'] == 'resnet_basic_block':
        module.conv1 = wrap_convHOSVD_with_var_layer(module.conv1, cfg["SVD_var"], True, cfg["k_hosvd"])
        module.conv2 = wrap_convHOSVD_with_var_layer(module.conv2, cfg["SVD_var"], True, cfg["k_hosvd"])
    elif cfg['type'] == 'resnet_bottleneck_block':
        module.conv1 = wrap_convHOSVD_with_var_layer(module.conv1, cfg["SVD_var"], True, cfg["k_hosvd"])
        module.conv2 = wrap_convHOSVD_with_var_layer(module.conv2, cfg["SVD_var"], True, cfg["k_hosvd"])
        module.conv3 = wrap_convHOSVD_with_var_layer(module.conv3, cfg["SVD_var"], True, cfg["k_hosvd"])
    elif cfg['type'] == 'resnet_layer':
        assert isinstance(module, nn.Sequential)
        ccfg = deepcopy(cfg)
        ccfg['type'] = 'resnet_basic_block'
        for blk_idx, blk in enumerate(module):
            add_hosvd_with_var_filter(blk, ccfg)
    elif cfg['type'] == 'ConvNormActivation':
        module[0] = wrap_convHOSVD_with_var_layer(module[0], cfg["SVD_var"], True, cfg["k_hosvd"])
    elif cfg['type'] == 'InvertedResidual':
        count = len(module.conv)
        module.conv[-2] = wrap_convHOSVD_with_var_layer(module.conv[-2], cfg["SVD_var"], True, cfg["k_hosvd"])
        for i in range(count - 2):
            module.conv[i][0] = wrap_convHOSVD_with_var_layer(module.conv[i][0], cfg["SVD_var"], True, cfg["k_hosvd"])
    elif cfg['type'] == "MCUNetBlock":
        # if hasattr(module.mobile_inverted_conv, 'inverted_bottleneck'):
        if module.mobile_inverted_conv.inverted_bottleneck is not None:
            module.mobile_inverted_conv.inverted_bottleneck.conv = wrap_convHOSVD_with_var_layer(
                module.mobile_inverted_conv.inverted_bottleneck.conv, cfg["SVD_var"], True, cfg["k_hosvd"])
        # if hasattr(module.mobile_inverted_conv, 'depth_conv'):
        if module.mobile_inverted_conv.depth_conv is not None:
            module.mobile_inverted_conv.depth_conv.conv = wrap_convHOSVD_with_var_layer(
                module.mobile_inverted_conv.depth_conv.conv, cfg["SVD_var"], True, cfg["k_hosvd"])
        # if hasattr(module.mobile_inverted_conv, 'point_linear'):
        if module.mobile_inverted_conv.point_linear is not None:
            module.mobile_inverted_conv.point_linear.conv = wrap_convHOSVD_with_var_layer(
                module.mobile_inverted_conv.point_linear.conv, cfg["SVD_var"], True, cfg["k_hosvd"])
    elif cfg['type'] == 'conv':
        module = wrap_convHOSVD_with_var_layer(module, cfg["SVD_var"], True, cfg["k_hosvd"])
    else:
        raise NotImplementedError
    return module

def add_gkpd_hosvd_filter(module: nn.Module, cfg):
    if cfg['type'] == 'cbr':
        module.conv = wrap_conv_GKPD_HOSVD_layer(module.conv, cfg["SVD_var"], True, cfg["k_gkpd_hosvd_1"], cfg["k_gkpd_hosvd_2"])
    elif cfg['type'] == 'resnet_basic_block':
        module.conv1 = wrap_conv_GKPD_HOSVD_layer(module.conv1, cfg["SVD_var"], True, cfg["k_gkpd_hosvd_1"], cfg["k_gkpd_hosvd_2"])
        module.conv2 = wrap_conv_GKPD_HOSVD_layer(module.conv2, cfg["SVD_var"], True, cfg["k_gkpd_hosvd_1"], cfg["k_gkpd_hosvd_2"])
    elif cfg['type'] == 'resnet_bottleneck_block':
        module.conv1 = wrap_conv_GKPD_HOSVD_layer(module.conv1, cfg["SVD_var"], True, cfg["k_gkpd_hosvd_1"], cfg["k_gkpd_hosvd_2"])
        module.conv2 = wrap_conv_GKPD_HOSVD_layer(module.conv2, cfg["SVD_var"], True, cfg["k_gkpd_hosvd_1"], cfg["k_gkpd_hosvd_2"])
        module.conv3 = wrap_conv_GKPD_HOSVD_layer(module.conv3, cfg["SVD_var"], True, cfg["k_gkpd_hosvd_1"], cfg["k_gkpd_hosvd_2"])
    elif cfg['type'] == 'resnet_layer':
        assert isinstance(module, nn.Sequential)
        ccfg = deepcopy(cfg)
        ccfg['type'] = 'resnet_basic_block'
        for blk_idx, blk in enumerate(module):
            add_gkpd_hosvd_filter(blk, ccfg)
    elif cfg['type'] == 'ConvNormActivation':
        module[0] = wrap_conv_GKPD_HOSVD_layer(module[0], cfg["SVD_var"], True, cfg["k_gkpd_hosvd_1"], cfg["k_gkpd_hosvd_2"])
    elif cfg['type'] == 'InvertedResidual':
        count = len(module.conv)
        module.conv[-2] = wrap_conv_GKPD_HOSVD_layer(module.conv[-2], cfg["SVD_var"], True, cfg["k_gkpd_hosvd_1"], cfg["k_gkpd_hosvd_2"])
        for i in range(count - 2):
            module.conv[i][0] = wrap_conv_GKPD_HOSVD_layer(module.conv[i][0], cfg["SVD_var"], True, cfg["k_gkpd_hosvd_1"], cfg["k_gkpd_hosvd_2"])
    elif cfg['type'] == "MCUNetBlock":
        # if hasattr(module.mobile_inverted_conv, 'inverted_bottleneck'):
        if module.mobile_inverted_conv.inverted_bottleneck is not None:
            module.mobile_inverted_conv.inverted_bottleneck.conv = wrap_conv_GKPD_HOSVD_layer(
                module.mobile_inverted_conv.inverted_bottleneck.conv, cfg["SVD_var"], True, cfg["k_gkpd_hosvd_1"], cfg["k_gkpd_hosvd_2"])
        # if hasattr(module.mobile_inverted_conv, 'depth_conv'):
        if module.mobile_inverted_conv.depth_conv is not None:
            module.mobile_inverted_conv.depth_conv.conv = wrap_conv_GKPD_HOSVD_layer(
                module.mobile_inverted_conv.depth_conv.conv, cfg["SVD_var"], True, cfg["k_gkpd_hosvd_1"], cfg["k_gkpd_hosvd_2"])
        # if hasattr(module.mobile_inverted_conv, 'point_linear'):
        if module.mobile_inverted_conv.point_linear is not None:
            module.mobile_inverted_conv.point_linear.conv = wrap_conv_GKPD_HOSVD_layer(
                module.mobile_inverted_conv.point_linear.conv, cfg["SVD_var"], True, cfg["k_gkpd_hosvd_1"], cfg["k_gkpd_hosvd_2"])
    elif cfg['type'] == 'conv':
        module = wrap_conv_GKPD_HOSVD_layer(module, cfg["SVD_var"], True, cfg["k_gkpd_hosvd_1"], cfg["k_gkpd_hosvd_2"])
    else:
        raise NotImplementedError
    return module
####################################################################

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

def register_SVD_with_var(module, cfgs):
    # filter_install_cfgs = cfgs['filter_install']
    logging.info("Registering Filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["cfgs"]: # Dò tên của từng conv layer sẽ bị cài filter
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_svd_with_var_filter(target, cfgs)

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_conv_batch(module, cfgs):
    # filter_install_cfgs = cfgs['filter_install']
    logging.info("Registering Filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["cfgs"]: # Dò tên của từng conv layer sẽ bị cài filter
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_avg_batch_layer(target, cfgs)

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_HOSVD_with_var(module, cfgs):
    # filter_install_cfgs = cfgs['filter_install']
    logging.info("Registering Filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["cfgs"]: # Dò tên của từng conv layer sẽ bị cài filter
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_hosvd_with_var_filter(target, cfgs)

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_GKPD_HOSVD(module, cfgs):
    # filter_install_cfgs = cfgs['filter_install']
    logging.info("Registering Filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["cfgs"]: # Dò tên của từng conv layer sẽ bị cài filter
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_gkpd_hosvd_filter(target, cfgs)

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)