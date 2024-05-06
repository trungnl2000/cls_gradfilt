# import torch
# import time

# # Tạo tensor X với kích thước 128x160x7x7
# X = torch.randn(128, 576, 4, 4).cuda()




# # Đo thời gian thực hiện torch.linalg.svd
# start_time = time.time()
# _ = torch.linalg.svd(X, full_matrices=False, driver='gesvdj')
# end_time = time.time()
# linalg_svd_time = end_time - start_time
# print("Thời gian thực hiện gesvdj:", linalg_svd_time)

# # Đo thời gian thực hiện torch.linalg.svd
# start_time = time.time()
# _ = torch.linalg.svd(X, full_matrices=False, driver='gesvda')
# end_time = time.time()
# linalg_svd_time = end_time - start_time
# print("Thời gian thực hiện gesvda:", linalg_svd_time)

# # Đo thời gian thực hiện torch.linalg.svd
# start_time = time.time()
# _ = torch.linalg.svd(X, full_matrices=False, driver='gesvd')
# end_time = time.time()
# linalg_svd_time = end_time - start_time
# print("Thời gian thực hiện gesvd:", linalg_svd_time)




import torch

def calculate_tensor_memory(tensor):
    element_size = tensor.element_size()  # kích thước của mỗi phần tử trong tensor (tính bằng byte)
    num_elements = tensor.numel()  # số lượng phần tử trong tensor
    total_memory = element_size * num_elements  # tổng bộ nhớ cần thiết
    return total_memory

def unfolding(n, A):
    shape = A.shape
    size = torch.prod(torch.tensor(shape))
    lsize = size // shape[n]
    sizelist = list(range(len(shape)))
    sizelist[n] = 0
    sizelist[0] = n
    return A.permute(sizelist).reshape(shape[n], lsize)

def truncated_svd(X, var, driver):
    # # X là tensor 2 chiều
    # U, S, V = torch.svd(X, some=True)
    U, S, V = torch.linalg.svd(X, full_matrices=False, driver=driver)
    total_variance = torch.sum(S**2)

    explained_variance = torch.cumsum(S**2, dim=0) / total_variance
    k = (explained_variance >= var).nonzero()[0].item() + 1
    # k=2
    return U[:, :k], S[:k], V[:, :k]

def modalsvd(n, A, var, driver):
    nA = unfolding(n, A)
    # return torch.svd(nA)
    return truncated_svd(nA, var, driver)

def hosvd(A, var, driver):
    Ulist = []
    # Vlist = []
    S = A.clone()

    # u0, _, _ = modalsvd(0, A, var)
    # S = torch.tensordot(S, u0, dims=([0], [0]))

    # u1, _, _ = modalsvd(1, A, var)
    # S = torch.tensordot(S, u1, dims=([0], [0]))

    # u2, _, _ = modalsvd(2, A, var)
    # S = torch.tensordot(S, u2, dims=([0], [0]))

    # u3, _, _ = modalsvd(3, A, var)
    # S = torch.tensordot(S, u3, dims=([0], [0]))
    # return S, u0, u1, u2, u3

    for i, ni in enumerate(A.shape):
        u, _, _ = modalsvd(i, A, var, driver)
        Ulist.append(u)
        # Vlist.append(v)
        S = torch.tensordot(S, u, dims=([0], [0]))
    # # return S, Ulist
    u0, u1, u2, u3 = Ulist[0], Ulist[1], Ulist[2], Ulist[3]
    # v0, v1, v2, v3 = Vlist[0], Vlist[1], Vlist[2], Vlist[3]

    return S, u0, u1, u2, u3
    # return S, v0, v1, v2, v3


A = torch.rand(128, 576, 4, 4).cuda()

import time

start_time = time.time()
S, u0, u1, u2, u3 = hosvd(A, 0.9, driver='gesvda')
end_time = time.time()
linalg_svd_time = end_time - start_time
print("Thời gian thực hiện gesvda:", linalg_svd_time)

start_time = time.time()
S, u0, u1, u2, u3 = hosvd(A, 0.9, driver='gesvdj')
end_time = time.time()
linalg_svd_time = end_time - start_time
print("Thời gian thực hiện gesvdj:", linalg_svd_time)



start_time = time.time()
S, u0, u1, u2, u3 = hosvd(A, 0.9, driver='gesvd')
end_time = time.time()
linalg_svd_time = end_time - start_time
print("Thời gian thực hiện gesvd:", linalg_svd_time)

start_time = time.time()
S, u0, u1, u2, u3 = hosvd(A, 0.9, driver=None)
end_time = time.time()
linalg_svd_time = end_time - start_time
print("Thời gian thực hiện None:", linalg_svd_time)