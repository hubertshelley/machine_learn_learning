import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 获取培训设备
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# 定义神经网络类
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
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


# 我们创建一个NeuralNetwork的实例，并将其移动到device，并打印其结构。
model = NeuralNetwork().to(device)
print(model)

# 要使用模型，我们将输入数据传递给它。这会执行模型的forward，以及一些后台操作。不要直接调用model.forward()！
#
# 在输入上调用模型返回一个二维张量，dim=0对应于每个类的10个原始预测值的输出，dim=1对应于每个输出的单个值。我们通过nn.Softmax模块的实例来获得预测概率。
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# 模型层
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# 初始化nn.Flatten层，将每个2D 28x28图像转换为784个像素值的连续数组（保持minibatch维度（在dim=0处）
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# 线性层是一个使用其存储权重和偏置在输入上应用线性变换的模块
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# 非线性激活是在模型的输入和输出之间创建复杂映射的原因。
# 它们在线性变换后应用，引入非线性，帮助神经网络学习各种现象
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential是一个有序的模块容器。数据以定义的相同顺序通过所有模块。
# 您可以使用顺序容器来组合一个快速网络，如seq_modules
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# 神经网络的最后一个线性层返回logits-[-infty, infty]中的原始值-这些值传递给nn.Softmax模块。
# 逻辑被缩放为值[0, 1]，表示模型对每个类的预测概率。昏暗参数表示值必须相加为1的维度
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# 模型参数
# 神经网络内的许多层都是参数化的，即在训练期间优化的相关权重和偏差。
# 子类nn.Module会自动跟踪模型对象内定义的所有字段，并使用模型的parameters()或named_parameters()方法访问所有参数。
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
