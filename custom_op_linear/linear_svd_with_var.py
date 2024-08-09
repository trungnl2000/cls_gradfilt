import torch as th
import torch.nn as nn
from torch.autograd import Function
from torch.nn.functional import linear

###### SVD by choosing principle components based on variance

# Cho 2 chiều
def truncated_svd(X, var=0.9, dim=0, driver=None):
    # dim là số chiều mà mình sẽ svd theo
    if(X.dim() != 2):
        n_samples, n_features = th.prod(th.tensor(X.shape[:dim+1])), th.prod(th.tensor(X.shape[dim+1:]))
        X_reshaped = X.view(n_samples, n_features)
    else:
        X_reshaped = X
    U, S, Vt = th.linalg.svd(X_reshaped, full_matrices=False, driver=driver)
    total_variance = th.sum(S**2)

    explained_variance = th.cumsum(S**2, dim=0) / total_variance
    # k = (explained_variance >= var).nonzero()[0].item() + 1
    # print("explained_variance: ", explained_variance)
    # print("k: ", k)
    nonzero_indices = (explained_variance >= var).nonzero()
    if len(nonzero_indices) > 0:
        # Nếu có ít nhất một phần tử >= var
        k = nonzero_indices[0].item() + 1
    else:
        # Nếu không có phần tử nào >= var, gán k bằng vị trí của phần tử lớn nhất
        k = explained_variance.argmax().item() + 1
    return th.matmul(U[:, :k], th.diag_embed(S[:k])) , Vt[:k, :]

def restore_tensor(Uk_Sk, Vk_t, shape):
    reconstructed_matrix = th.matmul(Uk_Sk, Vk_t)
    shape = tuple(shape)
    return reconstructed_matrix.view(shape)
#############################
class Linear_op(Function):
    @staticmethod
    # def forward(ctx, input, weight, bias=None, var=0.9):
    def forward(ctx, *args):
        input, weight, bias, var, svd_size = args

        output = th.matmul(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        # output = linear(input, weight, bias)

        input_Uk_Sk, input_Vk_t = truncated_svd(input, var=var)

        svd_size.append(th.tensor([input_Uk_Sk.shape[0], input_Vk_t.shape[0], input_Vk_t.shape[1]], device=input_Uk_Sk.device))

        ctx.save_for_backward(input_Uk_Sk, input_Vk_t, th.tensor(input.shape), weight, bias)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # input, weight, bias = ctx.saved_tensors
        input_Uk_Sk, input_Vk_t, input_shape, weight, bias = ctx.saved_tensors
        input = restore_tensor(input_Uk_Sk, input_Vk_t, input_shape)

            
    
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.mm(weight)
            grad_input = th.matmul(grad_output, weight)

        if ctx.needs_input_grad[1]:
            if (grad_output.dim() == 4):
                grad_weight = th.matmul(grad_output.permute(0, 1, 3, 2), input)
            elif (grad_output.dim() == 2):
                # grad_weight = grad_output.t().mm(input)
                grad_weight = th.matmul(grad_output.t(), input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias, None, None

class Linear_SVD(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            activate=False,
            var=0.9,
            svd_size=None):
        super(Linear_SVD, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.activate = activate
        self.var = var
        self.svd_size = svd_size

    def forward(self, input):
        if self.activate:
            output = Linear_op.apply(input, self.weight, self.bias, self.var, self.svd_size)
        else:
            output = super().forward(input)
        return output
    

def wrap_linear_svd_layer(linear, SVD_var, active, svd_size):
    has_bias = (linear.bias is not None)
    new_linear = Linear_SVD(in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=has_bias,
                        activate=active,
                        var=SVD_var,
                        svd_size=svd_size
                        )
    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear