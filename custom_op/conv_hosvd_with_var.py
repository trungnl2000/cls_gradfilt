import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d, pad
import torch.nn as nn

###### HOSVD base on variance #############

def unfolding(n, A):
    shape = A.shape
    size = th.prod(th.tensor(shape))
    lsize = size // shape[n]
    sizelist = list(range(len(shape)))
    sizelist[n] = 0
    sizelist[0] = n
    return A.permute(sizelist).reshape(shape[n], lsize)

def truncated_svd(X, var=0.9, driver=None):
    # X is 2D tensor
    U, S, Vt = th.linalg.svd(X, full_matrices=False, driver=driver)
    total_variance = th.sum(S**2)

    explained_variance = th.cumsum(S**2, dim=0) / total_variance
    # k = (explained_variance >= var).nonzero()[0].item() + 1
    nonzero_indices = (explained_variance >= var).nonzero()
    if len(nonzero_indices) > 0:
        # Nếu có ít nhất một phần tử >= var
        k = nonzero_indices[0].item() + 1
    else:
        # Nếu không có phần tử nào >= var, gán k bằng vị trí của phần tử lớn nhất
        k = explained_variance.argmax().item() + 1
    return U[:, :k], S[:k], Vt[:k, :]

def modalsvd(n, A, var, driver):
    nA = unfolding(n, A)
    # return torch.svd(nA)
    return truncated_svd(nA, var, driver)

def hosvd(A, var=0.9, driver=None):
    S = A.clone()
    u_list = []
    for i, ni in enumerate(A.shape):
        u, _, _ = modalsvd(i, A, var, driver)
        S = th.tensordot(S, u, dims=([0], [0]))
        u_list.append(u)
    return S, u_list

def restore_hosvd(S, u_list):
    restored_tensor = S.clone()
    for u in u_list:
        restored_tensor = th.tensordot(restored_tensor, u.t(), dims=([0], [0]))
    return restored_tensor

###############################################################
class Conv2dHOSVDop_with_var(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, weight, bias, stride, dilation, padding, groups, var, k_hosvd = args
        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups)

        S, u_list = hosvd(input, var=var)
        u0, u1, u2, u3 = u_list # B, C, H, W

        for idx in range(4):
            k_hosvd[idx].append(u_list[idx].shape[1])
        k_hosvd[4].append(input.shape)
        ctx.save_for_backward(S, u0, u1, u2, u3, weight, bias)

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        S, u0, u1, u2, u3, weight, bias  = ctx.saved_tensors
        B, C, H, W = u0.shape[0], u1.shape[0], u2.shape[0], u3.shape[0]
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None
        grad_output, = grad_outputs
        
        if ctx.needs_input_grad[0]:
            grad_input = nn.grad.conv2d_input((B,C,H,W), weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            _, _, K_H, K_W = weight.shape
            _, C_prime, H_prime, W_prime = grad_output.shape
            # Pad the input
            u2_padded = pad(u2, (0, 0, padding[0], padding[0]))
            u3_padded = pad(u3, (0, 0, padding[0], padding[0]))

            # Calculate Z1: (conv1x1)
            Z1 = conv2d(grad_output.permute(1,0,2,3), u0.T.unsqueeze(-1).unsqueeze(-1), groups=1).permute(1, 0, 2, 3) # Shape (C', B, H', W') conv with (K0, B', 1, 1) -> (C', K0, H', W') -> (K0, C', H', W')
            #______________________________________________________________________________________________________________
            # Calculate Z2: (conv1x1)
            Z2 = conv2d(S.permute(3, 2, 0, 1), u2_padded.unsqueeze(-1).unsqueeze(-1), groups=1).permute(2, 3, 1, 0) # Shape (K3, K2, K0, K1) conv with (H_padded, K2, 1, 1) -> (K3, H_padded, K0, K1) -> (K0, K1, H_padded, K3)
            #______________________________________________________________________________________________________________
            # Calculate Z3: (conv1x1)
            Z3 = conv2d(Z2.permute(2, 3, 0, 1), u3_padded.unsqueeze(-1).unsqueeze(-1), groups=1).permute(2, 3, 0, 1) # Shape (H_padded, K3, K0, K1) conv with (W_padded, K3, 1, 1) -> (H_padded, W_padded, K0, K1) -> (K0, K1, H_padded, W_padded)
            #______________________________________________________________________________________________________________
            # Calculate Z4 (conv2d)
            Z4 = conv2d(Z3.permute(1, 0, 2, 3), Z1.permute(1, 0, 2, 3), groups=1).permute(1, 0, 2, 3) # Shape (K1, K0, H_padded, W_padded) conv with (C', K0, H', W') --> (K1, C', K_H, K_W) -> (C', K1, K_H, K_W)
            #______________________________________________________________________________________________________________
            # calculate grad_weight
            if groups == C == C_prime:
                grad_weight = th.einsum("ckhw,ck->ckhw", Z4, u1).sum(dim=1).unsqueeze(1) # C', 1, K_H, K_W
            elif groups == 1:
                grad_weight = conv2d(Z4, u1.unsqueeze(-1).unsqueeze(-1), groups=1) # C', K1, K_H, K_W conv with C, K1, 1, 1 -> C', C, K_H, K_W
            else: # Havent tensorlize
                print("Havent optimized")

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

class Conv2dHOSVD_with_var(nn.Conv2d):
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
            k_hosvd = None
    ) -> None:
        if kernel_size is int:
            kernel_size = [kernel_size, kernel_size]
        if padding is int:
            padding = [padding, padding]
        if dilation is int:
            dilation = [dilation, dilation]
        # assert padding[0] == kernel_size[0] // 2 and padding[1] == kernel_size[1] // 2
        super(Conv2dHOSVD_with_var, self).__init__(in_channels=in_channels,
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
        self.k_hosvd = k_hosvd

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x, weight, bias, stride, padding, order, groups = args
        if self.activate:
            y = Conv2dHOSVDop_with_var.apply(x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, self.var, self.k_hosvd)
        else:
            y = super().forward(x)
        return y

def wrap_convHOSVD_with_var_layer(conv, SVD_var, active, k_hosvd):
    new_conv = Conv2dHOSVD_with_var(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         dilation=conv.dilation,
                         bias=conv.bias is not None,
                         groups=conv.groups,
                         padding=conv.padding,
                         activate=active,
                         var=SVD_var,
                         k_hosvd = k_hosvd
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv