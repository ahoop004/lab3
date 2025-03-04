import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LeNet(nn.Module):
    def __init__(self,num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(1, -1),
            nn.Linear(8 * 8 * 16, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.features(x)
        out = self.fc3(out)

        return out


class PretrainedAlexNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, cifar_mode=True):
        super(PretrainedAlexNet, self).__init__()
        self.model = models.alexnet(weights="DEFAULT") if pretrained else models.alexnet(weights=None)
        
        if cifar_mode:
            self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)


class PretrainedGoogLeNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, aux_logits=True, cifar_mode=True):
        super(PretrainedGoogLeNet, self).__init__()
        self.model = models.googlenet(weights="DEFAULT", aux_logits=aux_logits)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        if self.model.training and self.model.aux_logits:
            output, aux1, aux2 = self.model(x)
            return output, aux1, aux2
        else:
            return self.model(x)


class PretrainedResNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, resnet_version='resnet18',cifar_mode=True):

        super(PretrainedResNet, self).__init__()
        if resnet_version == 'resnet18':
            self.model = models.resnet18(weights="DEFAULT")
        elif resnet_version == 'resnet34':
            self.model = models.resnet34(weights="DEFAULT")
        elif resnet_version == 'resnet50':
            self.model = models.resnet50(weights="DEFAULT")
        elif resnet_version == 'resnet101':
            self.model = models.resnet101(weights="DEFAULT")
        elif resnet_version == 'resnet152':
            self.model = models.resnet152(weights="DEFAULT")
        else:
            raise ValueError("Invalid resnet_version specified. Choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.")
        
        if cifar_mode:
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.maxpool = nn.Identity()
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


