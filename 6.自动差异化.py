import torch

# 为了计算这些梯度，PyTorch有一个内置的微分引擎，称为torch.autograd。它支持自动计算任何计算图的梯度。
# 考虑最简单的单层神经网络，具有输入x、参数w和b以及一些损失函数。它可以在PyTorch中以以下方式定义
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# 计算梯度
loss.backward()
print(w.grad)
print(b.grad)

# 禁用跟踪计算(测试或使用模型时使用)
# 默认情况下，所有具有requires_grad=True张量都在跟踪其计算历史并支持梯度计算。
# 然而，在某些情况下，我们不需要这样做，
# 例如，当我们训练了模型并只想将其应用于一些输入数据时，即我们只想通过网络进行前向计算。
# 我们可以通过用torch.no_grad()块包围我们的计算代码来停止跟踪计算
z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)
# 实现相同结果的另一种方法是在张量上使用detach()方法
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

# 张量梯度和雅可比积
inp = torch.eye(4, 5, requires_grad=True)
out = (inp + 1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
