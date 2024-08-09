from typing import Union
import torch
import math

def multidimensional_unfold(tensor: torch.Tensor, kernel_size: torch.Tensor, stride: tuple,
                            device: torch.device = torch.device('cpu')) -> torch.Tensor:
    r"""Unfolds `tensor` by extracting patches of shape `kernel_size`.

    Reshaping and traversal for patch extraction both follow C-order convention (last index changes the fastest).

    Args:
        tensor: Input tensor to be unfolded with shape [N, *spatial_dims] (N is batch dimension)
        kernel_size: Patch size.
        stride: Stride of multidimensional traversal.
        device: Device used for operations.

    Returns:
       Unfolded tensor with shape [N, :math:`\prod_k kernel_size[k]`, L]

    """

    s_dims = tensor.shape[1:]  # spatial dimensions
    # Number of positions along each axis
    num_positions = (torch.tensor(s_dims) - kernel_size) // stride + 1

   
    # Start indices for each position in each axis
    positions = [torch.tensor([n * stride[i] for n in range(num_positions[i] - 1, -1, -1)]) for i in range(len(num_positions))]

    # Each column is a flattened patch
    output = torch.zeros(tensor.size(0), torch.prod(kernel_size).item(), 
                         torch.prod(num_positions).item(), device=device)
    
    for i, pos in enumerate(torch.cartesian_prod(*positions)):
        start_pos = torch.tensor([0, *pos])
        end_pos = torch.tensor([tensor.size(0), *(pos + kernel_size)])
        patch = multidimensional_slice(tensor, start_pos, end_pos)  # n,f2,c2,h2,w2
        output[:, :, torch.prod(num_positions).item() - 1 - i] = patch.reshape(tensor.size(0), -1)

    return output

def multidimensional_slice(tensor: torch.Tensor, start: torch.Tensor, stop: torch.Tensor) -> torch.Tensor:
    """Returns A[start_1:stop_1, ..., start_n:stop_n] for tensor A"

    Args:
        tensor: Input tensor `A`
        start: start indices
        stop: stop indices

    Returns:
         A[start_1:stop_1, ..., start_n:stop_n]
    """
    slices = tuple(slice(int(start[i]), int(stop[i])) for i in range(len(start)))
    return tensor[slices]


def kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Kronecker product between factors `a` and `b`

    Args:
        a: First factor
        b: Second factor

    Returns:
        Tensor containing kronecker product between `a` and `b`
    """
    return torch.einsum("rabcd,refgh->raebfcgdh", a, b).view(a.size(0), a.size(1)*b.size(1), a.size(2)*b.size(2),  a.size(3)*b.size(3), a.size(4)*b.size(4)).sum(dim=0)

def calculate_k(S, var):
    total_variance = torch.sum(S**2)
    explained_variance = torch.cumsum(S**2, dim=0) / total_variance
    nonzero_indices = (explained_variance >= var).nonzero()
    if len(nonzero_indices) > 0:
        k = nonzero_indices[0].item() + 1
    else:
        k = explained_variance.argmax().item() + 1
    return k


def gkpd(tensor: torch.Tensor, a_shape: Union[list, tuple], b_shape: Union[list, tuple], var: float) -> tuple:
    """Finds Kronecker decomposition of `tensor` via SVD.
    Patch traversal and reshaping operations all follow a C-order convention (last dimension changing fastest).
    Args:
        tensor (torch.Tensor): Tensor to be decomposed.
        a_shape (list, tuple): Shape of first Kronecker factor.
        b_shape (list, tuple): Shape of second Kronecker factor.
        var (float): Explained variance threshold to detect truncated rank

    Returns:
        a_hat: [rank, *a_shape]
        b_hat: [rank, *b_shape]
    """

    with torch.no_grad():
        w_unf = multidimensional_unfold(
            tensor.unsqueeze(0), kernel_size=b_shape, stride=b_shape, device=tensor.device
        )[0].T  # [num_positions, prod(s_dims)]
        u, s, v = torch.svd(w_unf)

        rank = calculate_k(s, var)
        s = s[:rank]
        u = u[:, :rank]
        vT = v.T[:rank, :]


        # Note: pytorch reshaping follows C-order as well
        # a_hat = torch.stack([s[i].item() * u[:, i].reshape(*a_shape) for i in range(rank)])  # [rank, *a_shape]
        # b_hat = torch.stack([v.T[i].reshape(*b_shape) for i in range(rank)])  # [rank, *b_shape]
        
        u_expanded = u.T.reshape(u.shape[1], *a_shape)
        # a_hat = s.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0) * u_expanded # Tương đương dùng einsum, nhưng eisum hiệu quả hơn
        a_hat = torch.einsum("r,rabcd->rabcd", s, u_expanded) # [rank, *a_shape]
        b_hat = vT.reshape(vT.shape[0], *b_shape)  # [rank, *b_shape]



    return a_hat, b_hat

def find_best_factor_shape(a_tensor):
    shape1 = torch.zeros_like(a_tensor)
    shape2 = torch.zeros_like(a_tensor)

    for i, a in enumerate(a_tensor):
        min_sum = a + 1
        best_x = None
        best_y = None

        for x in range(1, int(a.sqrt().item()) + 1):
            if a % x == 0:
                y = a // x
                current_sum = x + y
                if current_sum <= min_sum:
                    min_sum = current_sum
                    best_x = x
                    best_y = y
        shape1[i] = best_x
        shape2[i] = best_y
    return shape1, shape2