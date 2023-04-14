import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    # FashionMNIST功能为PIL图像格式，标签为整数。对于训练，我们需要将特征作为标准化张量，将标签作为单热编码张量。为了进行这些转换，我们使用ToTensor和Lambda。
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), src=1))
)
print(ds)
