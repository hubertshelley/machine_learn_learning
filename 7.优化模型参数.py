import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

# 超参数
# epochs - 在数据集上迭代的次数
# batch_size - 在参数更新之前通过网络传播的数据样本数量
# learning_rate - 每个批次/时代的模型参数更新多少。
#                 较小的值会产生缓慢的学习速度，而较大的值可能会导致训练期间的不可预测行为
learning_rate = 1e-3
batch_size = 64
epochs = 5

# 优化循环
# 一旦我们设置了超参数，我们就可以通过优化循环训练和优化我们的模型。优化循环的每个迭代都被称为纪元。#
# 每个时代由两个主要部分组成：
#   训练循环 - 迭代训练数据集，并尝试收敛到最优参数。
#   测试循环 - 迭代测试数据集，以检查模型性能是否正在提高。

# 损失功能
# 当看到一些训练数据时，我们未经训练的网络可能不会给出正确的答案。
# 损失函数衡量所获得的结果与目标值的差异程度，这是我们希望在训练期间最小化的损失函数。
# 为了计算损失，我们使用给定数据样本的输入进行预测，并将其与真实的数据标签值进行比较。
# 常见的损失函数包括用于回归任务的nn.MSELoss（均方误差）和用于分类的nn.NLLLoss（负对数可能性）。
# nn.CrossEntropyLoss结合了nn.LogSoftmax和nn.NLLLoss。
# 我们将模型的输出日志传递给nn.CrossEntropyLoss，这将使日志正常化并计算预测错误。
# 初始化损失方法
loss_fn = nn.CrossEntropyLoss()

# 优化器
# 优化是调整模型参数以减少每个训练步骤中的模型误差的过程。
# 优化算法定义了如何执行此过程（在本例中，我们使用随机梯度下降）。
# 所有优化逻辑都封装在optimizer对象中。
# 在这里，我们使用SGD优化器；此外，PyTorch中有许多不同的优化器，如ADAM和RMSProp，它们更适用于不同类型的模型和数据。
# 我们通过注册需要训练的模型参数并传递学习率超参数来初始化优化器。
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 全面实施
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
