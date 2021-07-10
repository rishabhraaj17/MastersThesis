import pytorchvideo.models
import torch

from src_lib.models_hub.spatio_temporal.utils import PackPathway

if __name__ == '__main__':
    # net = ResNet3d(depth=18, pretrained=None)
    net = pytorchvideo.models.create_slowfast()

    # inp = torch.rand((2, 3, 2, 720, 480))
    packer = PackPathway()
    inp = packer(torch.rand((2, 3, 32, 240, 240)))
    o = net(inp)
    print()
