#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import pdb

import numpy as np
from torchvision import datasets, transforms

def split_and_shuffle(dataset, train_ratio):
    idxs = np.arange(len(dataset))

    if train_ratio <= 0 or train_ratio >= 100:
        raise ValueError("Percentage should be between 0 and 100 (exclusive)")

    # Lấy giá trị seed hiện tại (được set theo seed_everything_default)
    current_seed = np.random.get_state()[1][0]
    # print("Current Seed:", current_seed)
    # Thiết lập seed cho random
    np.random.seed(current_seed)

    # Trộn mảng đầu vào
    np.random.shuffle(idxs)

    # Thiết lập lại seed cho random vì lúc này seed sẽ lại đổi
    np.random.seed(current_seed)

    # Số lượng phần tử trong mỗi phần
    num_elements = len(idxs)
    num_elements_part1 = int(num_elements * train_ratio)

    # Chia mảng thành hai phần
    idxs_part1 = idxs[:num_elements_part1]
    idxs_part2 = idxs[num_elements_part1:]

    return idxs_part1, idxs_part2
