import torch
import torchvision.models as models

model = models.vgg16(weights='IMAGENET1K_V1')
# 保存模型
torch.save(model.state_dict(), 'model_weights.pth')

# 加载模型
model = models.vgg16()  # we do not specify weights, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# 保存和加载带有形状的模型
# 保存模型
torch.save(model, 'model.pth')
# 加载模型
model = torch.load('model.pth')
