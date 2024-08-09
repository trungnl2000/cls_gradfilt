import torch as th

def unfolding(n, A):
    shape = A.shape
    size = th.prod(th.tensor(shape))
    lsize = size // shape[n]
    sizelist = list(range(len(shape)))
    sizelist[n] = 0
    sizelist[0] = n
    return A.permute(sizelist).reshape(shape[n], lsize)

def truncated_svd(X, var=0.9):
    # X is 2D tensor
    U, S, Vt = th.linalg.svd(X, full_matrices=False)
    total_variance = th.sum(S**2)

    explained_variance = th.cumsum(S**2, dim=0) / total_variance
    # print("explained_variance: ", explained_variance)
    nonzero_indices = (explained_variance >= var).nonzero()
    if len(nonzero_indices) > 0:
        # Nếu có ít nhất một phần tử >= var
        k = nonzero_indices[0].item() + 1
    else:
        # Nếu không có phần tử nào >= var, gán k bằng vị trí của phần tử lớn nhất
        k = explained_variance.argmax().item() + 1
    return U[:, :k], S[:k], Vt[:k, :]

def modalsvd(n, A, var):
    nA = unfolding(n, A)
    # return torch.svd(nA)
    return truncated_svd(nA, var)

def hosvd(A, var=0.9, skip_first_dim=False):
    S = A.clone()
    u_list = []
    if skip_first_dim:
        for i in range(1, A.dim()):
            u, _, _ = modalsvd(i, A, var)
            S = th.tensordot(S, u, dims=([1], [0]))
            u_list.append(u)
    else:
        for i in range(A.dim()):
            u, _, _ = modalsvd(i, A, var)
            S = th.tensordot(S, u, dims=([0], [0]))
            u_list.append(u)
    return S, u_list

def restore_hosvd(S, u_list, skip_first_dim=False):
    restored_tensor = S.clone()
    for u in u_list:
        if skip_first_dim:
            restored_tensor = th.tensordot(restored_tensor, u.t(), dims=([1], [0]))
        else:
            restored_tensor = th.tensordot(restored_tensor, u.t(), dims=([0], [0]))

    return restored_tensor