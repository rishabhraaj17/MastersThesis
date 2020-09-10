from typing import Any

import torch
from torch.hub import load_state_dict_from_url
from torchvision.models.vgg import vgg13_bn, vgg19_bn, vgg11_bn
from torchvision.models.resnet import ResNet, Bottleneck, model_urls
from torchvision.models.densenet import densenet121
from torchvision.models.segmentation.segmentation import fcn_resnet50


def get_vgg_layer_activations(x, layer_number: int = 3):
    model = vgg19_bn(pretrained=True).features
    for m in model.children():
        m.requires_grad = False
    model.eval()

    out = model[:layer_number](x)
    return out


class ResNetFeatures(ResNet):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetFeatures, self).__init__(block, layers, num_classes=1000, zero_init_residual=False,
                                             groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                             norm_layer=None)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)  # torch.Size([1, 512, 244, 178])
        # x = self.layer3(x)  # torch.Size([1, 1024, 122, 89])
        # x = self.layer4(x)  # torch.Size([1, 2048, 61, 45])
        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetFeatures(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def get_resnet_layer_activations(x):
    model = resnet101(pretrained=True)
    for m in model.children():
        m.requires_grad = False
    model.eval()

    out = model(x)
    return out


def get_densenet_layer_activations(x, layer_number: int = 3):
    model = densenet121(pretrained=True).features
    for m in model.children():
        m.requires_grad = False
    model.eval()

    out = model[:layer_number](x)
    return out


def get_densenet_filtered_layer_activations(x, layer_number: int = 3):
    model = densenet121(pretrained=True).features
    for m in model.children():
        m.requires_grad = False
    model.eval()

    out = model[:layer_number](x)
    out = model[4].denselayer1(out)
    return out


if __name__ == '__main__':
    inp = torch.randn((1, 3, 1945, 1422))
    # net = resnet101(pretrained=True)
    net = vgg11_bn(pretrained=False)
    print(net)
    o = net.features[:20](inp)  # 38
    o = o.size()
    print(o)

    # enc_1 = self.encoder.features[:6](x)
    # print(enc_1.size())
    # enc_2 = self.encoder.features[6:13](enc_1)
    # print(enc_2.size())
    # enc_3 = self.encoder.features[13:26](enc_2)
    # print(enc_3.size())
    # enc_4 = self.encoder.features[26:39](enc_3)
    # print(enc_4.size())
    #
    # dec_1 = self.decoder[:12](enc_4)
    # print(dec_1.size())
    # dec_2 = self.decoder[12:25](dec_1)
    # print(dec_2.size())
    # dec_3 = self.decoder[25:](dec_2)
    # print(dec_3.size())
