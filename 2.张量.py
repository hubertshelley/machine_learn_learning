import torch
import numpy as np

# 直接来自数据
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print("直接来自数据", x_data)

# 来自NumPy数组
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print("来自NumPy数组", x_np)

# 来自另一个张量
x_ones = torch.ones_like(x_data)  # 保留原属性
print(f"来自另一个张量：Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # 覆盖数据类型
print(f"来自另一个张量：Random Tensor: \n {x_rand} \n")

# 使用随机或常数值
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# 张量属性
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 张量操作
# 如果可用，我们将张量移至 GPU
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# 标准numpy样索引和切片
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

# 连接张量
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print("连接张量", t1)

# 算术运算
# 这计算两个张量之间的矩阵乘法。 y1, y2, y3 将具有相同的值 ``tensor.T`` 返回张量的转置
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print("算术运算:y1, y2, y3")
print(y1)
print(y2)
print(y3)

# 这将计算元素乘积。 z1、z2、z3 将具有相同的值
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print("算术运算:z1, z2, z3")
print(z1)
print(z2)
print(z3)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
