import torch
import torch.nn as nn
from torch.autograd import Function


class Linear_op(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias

class Linear(nn.Linear):
    def __init__(
            self,
            input_features,
            output_features,
            bias=True,
            device=None,
            dtype=None,
            activate=False):
        super(Linear, self).__init__(
            in_features=input_features,
            out_features=output_features,
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
    

def wrap_linear_layer(linear, active):
    new_linear = Linear(input_features=linear.input_features,
                        output_features=linear.output_features,
                        bias=linear.bias,
                        activate=active,
                         )
    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear