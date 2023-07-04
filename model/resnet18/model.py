from torchvision.models import ResNet18_Weights, resnet18
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=False)
        
    def forward(self, input1, input2):
        output1 = self.resnet18(input1)
        output2 = self.resnet18(input2)
        return output1, output2