from .gkpd import gkpd, kron, find_best_factor_shape
from .hosvd import hosvd, restore_hosvd

import torch
import time
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d, pad
import torch.nn as nn

###############################################################
class Conv2d_GKPD_HOSVD_op(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, weight, bias, stride, dilation, padding, groups, var, k_gkpd_hosvd_1, k_gkpd_hosvd_2 = args

        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups)

        a_shape, b_shape = find_best_factor_shape(torch.tensor(input.shape))

        a_hat, b_hat = gkpd(input, a_shape, b_shape, var=var)
        Sa, u_list_a = hosvd(a_hat, var=var, skip_first_dim=True)
        Sb, u_list_b = hosvd(b_hat, var=var, skip_first_dim=True)
        
        for idx in range(4):
            k_gkpd_hosvd_1[idx].append(u_list_a[idx].shape[1])
            k_gkpd_hosvd_2[idx].append(u_list_b[idx].shape[1])
        k_gkpd_hosvd_1[4].append(a_hat.shape)
        k_gkpd_hosvd_2[4].append(b_hat.shape)

        ctx.save_for_backward(Sa, u_list_a[0], u_list_a[1], u_list_a[2], u_list_a[3],
                              Sb, u_list_b[0], u_list_b[1], u_list_b[2], u_list_b[3],
                              weight, bias)

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:

        Sa, ua_0, ua_1, ua_2, ua_3,\
        Sb, ub_0, ub_1, ub_2, ub_3,\
        weight, bias  = ctx.saved_tensors
        

        a_hat = restore_hosvd(Sa, [ua_0, ua_1, ua_2, ua_3], skip_first_dim=True)
        b_hat = restore_hosvd(Sb, [ub_0, ub_1, ub_2, ub_3],  skip_first_dim=True)
        input = kron(a_hat, b_hat)

        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None
        grad_output, = grad_outputs
        
        if ctx.needs_input_grad[0]:
            grad_input = nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

class Conv2d_GKPD_HOSVD(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            padding=0,
            device=None,
            dtype=None,
            activate=False,
            var=1,
            k_gkpd_hosvd_1 = None,
            k_gkpd_hosvd_2 = None
    ) -> None:
        if kernel_size is int:
            kernel_size = [kernel_size, kernel_size]
        if padding is int:
            padding = [padding, padding]
        if dilation is int:
            dilation = [dilation, dilation]
        # assert padding[0] == kernel_size[0] // 2 and padding[1] == kernel_size[1] // 2
        super(Conv2d_GKPD_HOSVD, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding=padding,
                                        padding_mode='zeros',
                                        device=device,
                                        dtype=dtype)
        self.activate = activate
        self.var = var
        self.k_gkpd_hosvd_1 = k_gkpd_hosvd_1
        self.k_gkpd_hosvd_2 = k_gkpd_hosvd_2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x, weight, bias, stride, padding, order, groups = args
        if self.activate:
            y = Conv2d_GKPD_HOSVD_op.apply(x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, self.var, self.k_gkpd_hosvd_1, self.k_gkpd_hosvd_2)
        else:
            y = super().forward(x)
        return y

def wrap_conv_GKPD_HOSVD_layer(conv, SVD_var, active, k_gkpd_hosvd_1, k_gkpd_hosvd_2):
    new_conv = Conv2d_GKPD_HOSVD(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         dilation=conv.dilation,
                         bias=conv.bias is not None,
                         groups=conv.groups,
                         padding=conv.padding,
                         activate=active,
                         var=SVD_var,
                         k_gkpd_hosvd_1 = k_gkpd_hosvd_1,
                         k_gkpd_hosvd_2 = k_gkpd_hosvd_2
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv