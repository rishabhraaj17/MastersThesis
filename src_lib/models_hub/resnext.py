from functools import reduce

import torch.nn as nn
from torchvision.models import resnext50_32x4d, resnext101_32x8d


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, x):
        output = []
        for module in self._modules.values():
            output.append(module(x))
        return output if output else x


class Lambda(LambdaBase):
    def forward(self, x):
        return self.lambda_func(self.forward_prepare(x))


class LambdaMap(LambdaBase):
    def forward(self, x):
        return list(map(self.lambda_func, self.forward_prepare(x)))


class LambdaReduce(LambdaBase):
    def forward(self, x):
        return reduce(self.lambda_func, self.forward_prepare(x))


class ResNeXt101(nn.Module):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        net = resnext101()

        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3: 5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4


def _resnext_core(model, in_channels=3):
    model.conv1 = nn.Conv2d(in_channels, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias=False)
    model.avgpool = nn.AvgPool2d((7, 7), (1, 1))
    model.fc = nn.Sequential(
        Lambda(lambda x: x.view(x.size(0), -1)),
        Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x),
        nn.Linear(2048, 1000)
    )


def resnext50():
    model = resnext50_32x4d()
    _resnext_core(model)
    return model


def resnext101():
    model = resnext101_32x8d()
    _resnext_core(model)
    return model
