import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d
import torch.nn as nn
# from math import ceil

def truncated_svd(X, k):
    # X = X.to(device='cuda' if th.cuda.is_available() else 'cpu')
    U, S, Vt = th.linalg.svd(X)
    Uk = U[:, :, :, :k]
    Sk = S[:, :, :k]
    Vk_t = Vt[:, :, :k, :]
    # new_X = np.dot(Uk, np.dot(Sk, Vk))
    # return new_X
    return U, S, Vt, Uk, Sk, Vk_t

def reconstruct_from_svd(U, S, Vh):
    return th.matmul(th.matmul(U, th.diag(S)), Vh)

def calculate_tensor_memory(tensor):
    total_memory = tensor.element_size() * tensor.numel()  # tổng bộ nhớ cần thiết
    return total_memory

def calculate_error(Sk, S):
    Sk_ = th.reshape(Sk, (Sk.shape[0]*Sk.shape[1], Sk.shape[2]))
    S_ = th.reshape(S, (S.shape[0]*S.shape[1], S.shape[2]))
    error = 0
    for i in range(len(Sk_)):
        error += th.sum(Sk_[i]**2)/th.sum(S_[i]**2)
    return error/i

def restore_tensor(Uk, Sk, Vk_t):
    original_tensor = th.empty(Uk.shape[0], Uk.shape[1], Uk.shape[2], Vk_t.shape[-1], device=Uk.device)

    shape = Uk.shape[0:2] # Kích thước 2 chiều đầu của tensor cần khôi phục, nó bằng kích thước 2 chiều đầu của Uk
    for i in range(shape[0]):
        original_tensor1 = th.empty(Uk.shape[1], Uk.shape[2], Vk_t.shape[-1], device=Uk.device)
        for j in range(shape[1]):
            t = reconstruct_from_svd(Uk[i][j], Sk[i][j], Vk_t[i][j])
            original_tensor1[j] = t
        original_tensor[i] = original_tensor1
    return original_tensor

###############################################################
class Conv2dSVDop(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, weight, bias, stride, dilation, padding, groups = args

        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups) # Chỗ này như bình thường

        # # Phân rã svd ở đây
        input_U, input_S, input_Vt, input_Uk, input_Sk, input_Vk_t = truncated_svd(input, k=3)
        # weight_U, weight_S, weight_Vt, weight_Uk, weight_Sk, weight_Vk_t = truncated_svd(weight, k=3)

        # print("input: ", input.device.type, " ", input.device.index)
        # print("input_Uk: ", input_Uk.device.type, " ", input_Uk.device.index)


        # print(grad_weight.device.type) 
        # print(grad_weight.device.index)
        # device='cuda' if th.cuda.is_available() else 'cpu'
        # input_Uk = input_Uk.to(device)
        # input_Sk = input_Sk.to(device)
        # input_Vk_t = input_Vk_t.to(device)
        # weight_Uk = weight_Uk.to(device)
        # weight_Sk = weight_Sk.to(device)
        # weight_Vk_t = weight_Vk_t.to(device)





        ctx.save_for_backward(input_Uk, input_Sk, input_Vk_t, weight, bias)
        # ctx.save_for_backward(input_Uk, input_Sk, input_Vk_t, weight_Uk, weight_Sk, weight_Vk_t, bias)
        # ctx.save_for_backward(input, weight, bias)

        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # input, weight, bias = ctx.saved_tensors

        input_Uk, input_Sk, input_Vk_t, weight, bias = ctx.saved_tensors
        print("input_Uk: ", input_Uk.device.type, " ", input_Uk.device.index)

        # input_Uk, input_Sk, input_Vk_t, weight_Uk, weight_Sk, weight_Vk_t, bias = ctx.saved_tensors
        input = restore_tensor(input_Uk, input_Sk, input_Vk_t)
        # weight = restore_tensor(weight_Uk, weight_Sk, weight_Vk_t)
        # print("input: ", input.device.type, " ", input.device.index)

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
        return grad_input, grad_weight, grad_bias, None, None, None, None

class Conv2dSVD(nn.Conv2d):
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
            activate=False
    ) -> None:
        if kernel_size is int:
            kernel_size = [kernel_size, kernel_size]
        if padding is int:
            padding = [padding, padding]
        if dilation is int:
            dilation = [dilation, dilation]
        # assert padding[0] == kernel_size[0] // 2 and padding[1] == kernel_size[1] // 2
        super(Conv2dSVD, self).__init__(in_channels=in_channels,
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

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x, weight, bias, stride, padding, order, groups = args
        if self.activate:
            y = Conv2dSVDop.apply(x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups)
        else:
            y = super().forward(x)
        return y

def wrap_convSVD_layer(conv, active):
    new_conv = Conv2dSVD(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         dilation=conv.dilation,
                         bias=conv.bias is not None,
                         groups=conv.groups,
                         padding=conv.padding,
                         activate=active
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv