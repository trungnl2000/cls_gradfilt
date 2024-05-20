import logging
from copy import deepcopy
from functools import reduce
import torch.nn as nn
from .linear_hosvd_with_var import wrap_linear_hosvd_layer
from .linear_svd_with_var import wrap_linear_svd_layer
from .linear_avg_batch import wrap_linear_avg_batch



def register_HOSVD_with_var(module, cfgs):
    logging.info("Registering Filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["cfgs"]: # Dò tên của từng conv layer sẽ bị cài filter
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        upd_layer = wrap_linear_hosvd_layer(target, cfgs["SVD_var"], True)

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_avg_batch(module, cfgs):
    logging.info("Registering Filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["cfgs"]: # Dò tên của từng conv layer sẽ bị cài filter
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        upd_layer = wrap_linear_avg_batch(target, True)

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_SVD_with_var(module, cfgs):
    logging.info("Registering Filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["cfgs"]: # Dò tên của từng conv layer sẽ bị cài filter
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        upd_layer = wrap_linear_svd_layer(target, cfgs["SVD_var"], True)

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)