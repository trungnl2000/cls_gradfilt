import torch
import torch.nn as nn
from torch.autograd import Function

###### Avg batch #############

def linear_avg_batch(X):
    return torch.mean(X, dim=0)

def restore_tensor(X_avg_batch):
    return X_avg_batch.unsqueeze(0)

#############################
class Linear_op(Function):
    @staticmethod
    # def forward(ctx, input, weight, bias=None, var=0.9):
    def forward(ctx, *args):
        input, weight, bias = args

        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        # output = linear(input, weight, bias)

        input_avg_batch = linear_avg_batch(input)
        ctx.save_for_backward(input_avg_batch, torch.tensor(input.shape), weight, bias)
        # Tiếp tục triển khai cho trường hợp nếu ma trận là 2 chiều thì không cần avg
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_avg_batch, input_shape, weight, bias = ctx.saved_tensors
        # print("input_shape: ", input_shape)
        input = restore_tensor(input_avg_batch)
        grad_output_ = torch.mean(grad_output, dim=0).unsqueeze(0)
    
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.mm(weight)
            grad_input = torch.matmul(grad_output, weight)

        if ctx.needs_input_grad[1]:
            if (grad_output_.dim() == 4):
                grad_weight = torch.matmul(grad_output_.permute(0, 1, 3, 2), input)
            elif (grad_output_.dim() == 2):
                # grad_weight = grad_output.t().mm(input)
                grad_weight = grad_output_.t().mm(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias

class Linear_avg_batch(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            activate=False):
        super(Linear_avg_batch, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.activate = activate
    def forward(self, input):
        if self.activate:
            output = Linear_op.apply(input, self.weight, self.bias)
        else:
            output = super().forward(input)
        return output
    

def wrap_linear_avg_batch(linear, active):
    has_bias = (linear.bias is not None)
    new_linear = Linear_avg_batch(in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=has_bias,
                        activate=active
                        )
    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear