import torch
import torch.nn as nn
import torchvision.models as models

# 定义ResNet-18模型结构
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        # 加载预训练的ResNet-18模型
        self.resnet18 = models.resnet18(pretrained=True)
        # 移除最后一层分类头
        self.backbone = nn.Sequential(*list(self.resnet18.children())[:-2])
        # 添加一个自定义的分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)  # 修改num_classes以匹配你的任务
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x