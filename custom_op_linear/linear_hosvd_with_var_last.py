import torch
import torch.nn as nn
from torch.autograd import Function

###### HOSVD base on variance #############

def unfolding(n, A):
    shape = A.shape
    size = torch.prod(torch.tensor(shape))
    lsize = size // shape[n]
    sizelist = list(range(len(shape)))
    sizelist[n] = 0
    sizelist[0] = n
    return A.permute(sizelist).reshape(shape[n], lsize)

def truncated_svd(X, var=0.9, driver=None):
    # X is 2D tensor
    U, S, Vt = torch.linalg.svd(X, full_matrices=False, driver=driver)
    total_variance = torch.sum(S**2)

    explained_variance = torch.cumsum(S**2, dim=0) / total_variance
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
    
    u0, _, _ = modalsvd(0, A, var, driver)
    S = torch.tensordot(S, u0, dims=([0], [0]))

    u1, _, _ = modalsvd(1, A, var, driver)
    S = torch.tensordot(S, u1, dims=([0], [0]))
    
    return S, u0, u1

def restore_hosvd(S, u0, u1):
    # Initialize the restored tensor
    restored_tensor = S.clone()

    # Multiply each mode of the restored tensor by the corresponding U matrix
    restored_tensor = torch.tensordot(restored_tensor, u0.t(), dims=([0], [0]))
    restored_tensor = torch.tensordot(restored_tensor, u1.t(), dims=([0], [0]))
    return restored_tensor
#############################
class Linear_op_last(Function):
    @staticmethod
    # def forward(ctx, input, weight, bias=None, var=0.9):
    def forward(ctx, *args):
        input, weight, bias, var = args

        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        S, u0, u1 = hosvd(input, var=var)
        ctx.save_for_backward(S, u0, u1, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # input, weight, bias = ctx.saved_tensors
        S, U0, U1, weight, bias = ctx.saved_tensors
        input = restore_hosvd(S, U0, U1)
            
    
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.mm(weight)
            grad_input = torch.matmul(grad_output, weight)

        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output.t().mm(input)
            grad_weight = torch.matmul(grad_output.t(), input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias, None

class Linear_HOSVD_last(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            activate=False,
            var=0.9):
        super(Linear_HOSVD_last, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.activate = activate
        self.var = var

    def forward(self, input):
        if self.activate:
            output = Linear_op_last.apply(input, self.weight, self.bias, self.var)
        else:
            output = super().forward(input)
        return output
    

def wrap_linear_hosvd_layer_last(linear, SVD_var, active):
    has_bias = (linear.bias is not None)
    new_linear = Linear_HOSVD_last(in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=has_bias,
                        activate=active,
                        var=SVD_var
                        )
    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear