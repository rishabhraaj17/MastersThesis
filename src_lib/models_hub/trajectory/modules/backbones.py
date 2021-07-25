from typing import List, Dict

import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from src_lib.models_hub.trajectory.modules.frozenbn import FrozenBatchNorm2d
from src_lib.models_hub.trajectory.modules.pos_encoding import build_position_encoding
from src_lib.models_hub.trajectory.modules.utils import NestedTensor


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool,
                 return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (not train_backbone
                    or 'layer2' not in name
                    and 'layer3' not in name
                    and 'layer4' not in name):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [4, 8, 16, 32]
            self.num_channels = [256, 512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class BackboneV0(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 pretrained: bool = True):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=norm_layer)
        super().__init__(backbone, train_backbone,
                         return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Backbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 dilation: bool,
                 pretrained: bool = True):
        super().__init__()
        norm_layer = FrozenBatchNorm2d
        self.backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=norm_layer)

        for name, parameter in self.backbone.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)

    def forward(self, x):
        return self.backbone(x)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for x in xs.values():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone,
                        train_backbone,
                        return_interm_layers,
                        args.dilation)
    model = Joiner(backbone, position_embedding)
    return model


if __name__ == '__main__':
    m = Backbone('resnet18', train_backbone=True, dilation=False, pretrained=True)
    inp = torch.randn((2, 3, 64, 64))
    out = m(inp)
    print()
