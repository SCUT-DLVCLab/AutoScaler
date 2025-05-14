from torchvision.models.resnet import resnet18, ResNet18_Weights, resnet101, ResNet101_Weights
import torch

def _forward_reimpl(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

if __name__=='__main__':
    model=resnet18(weights=ResNet18_Weights.DEFAULT)
    data=torch.randn([12,3,128,256])
    model.forward=_forward_reimpl.__get__(model)
    print(model(data).shape)

