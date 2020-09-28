from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg13_bn, vgg19_bn, vgg11_bn
import matplotlib.pyplot as plt
import numpy as np
from pl_bolts.models.autoencoders import AE

vgg_decoder_arch = {'E': [512, 512, 512, 512, 'U', 256, 256, 256, 256, 'U', 128, 128, 'U', 64, 64],
                    'D': [512, 512, 512, 'U', 256, 256, 256, 'U', 128, 128, 'U', 64, 64],
                    'B': [512, 512, 'U', 256, 256, 'U', 128, 128, 'U', 64, 64],
                    'A': [512, 512, 'U', 256, 256, 'U', 128, 'U', 64]}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 512
    for v in cfg:
        if v == 'U':
            layers += [nn.UpsamplingBilinear2d(scale_factor=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGEncoder(nn.Module):
    def __init__(self, encoder, pretrained_encoder=True):
        super(VGGEncoder, self).__init__()
        self.encoder = encoder(pretrained=pretrained_encoder)

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def forward(self, x):
        # Arch 'A'
        enc_1 = self.encoder.features[:3](x)
        enc_2 = self.encoder.features[3:7](enc_1)
        enc_3 = self.encoder.features[7:14](enc_2)
        enc_4 = self.encoder.features[14:21](enc_3)

        return enc_4


class VGGDecoder(nn.Module):
    def __init__(self, decoder):
        super(VGGDecoder, self).__init__()
        self.decoder = decoder
        self.end_layers = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                        nn.Conv2d(32, 16, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                        nn.Conv2d(16, 3, kernel_size=3, padding=1))

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def forward(self, x, size):
        # Arch 'A'
        dec_1 = self.decoder[:6](x)  # + x
        dec_2 = self.decoder[6:13](dec_1)
        dec_3 = self.decoder[13:](dec_2)
        upsampled = F.upsample_bilinear(dec_3, size=size)

        out = self.end_layers(upsampled)

        return out


class VanillaAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, pretrained=False):
        super(VanillaAutoEncoder, self).__init__()
        self.encoder = VGGEncoder(encoder=encoder, pretrained_encoder=pretrained)
        self.decoder = VGGDecoder(decoder=decoder)

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        enc = self.encoder(x)
        dec = self.decoder(enc, size=(h, w))
        return dec


class UnsupervisedTP(nn.Module):
    def __init__(self, encoder, decoder, pretrained=False):
        super(UnsupervisedTP, self).__init__()
        self.encoder = VGGEncoder(encoder=encoder, pretrained_encoder=pretrained)
        self.decoder = VGGDecoder(decoder=decoder)
        self.position_map_processor = nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1),
                                                    nn.BatchNorm1d(512),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1),
                                                    nn.BatchNorm1d(128),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1),
                                                    nn.BatchNorm1d(32),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1),
                                                    nn.ReLU(inplace=True))
        self.conv = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1)

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def forward(self, x, position_map=None):
        h, w = x.size(2), x.size(3)
        if position_map is not None:
            x = torch.cat((x, position_map), dim=1)
            x = self.conv(x)
        enc = self.encoder(x)
        pos_map_1 = torch.cat((enc[0], enc[1]))
        pos_map_2 = torch.cat((enc[1], enc[2]))

        pos_maps = torch.stack((pos_map_1, pos_map_2))
        pos_maps = pos_maps.view(pos_maps.size(0), pos_maps.size(1), -1)
        pos_maps = self.position_map_processor(pos_maps).view(pos_maps.size(0), 1, pos_map_1.size(1), pos_map_1.size(2))

        pos_map_1 = pos_maps[0]
        pos_map_2 = pos_maps[1]

        avg = torch.stack((pos_map_1, pos_map_2)).mean(0).unsqueeze(0)

        dec_in = avg + enc[1].unsqueeze(0)

        dec = self.decoder(dec_in, size=(h, w))
        return dec, pos_map_1.unsqueeze(0), pos_map_2.unsqueeze(0)  # torch.Size([1, 3, 486, 355]),
        # torch.Size([1, 1, 60, 44]), torch.Size([1, 1, 60, 44])


class VGGEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, pretrained_encoder=True):
        super(VGGEncoderDecoder, self).__init__()
        self.encoder = encoder(pretrained=pretrained_encoder)
        self.decoder = decoder

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def forward(self, x):
        # Arch 'E'
        # enc_1 = self.encoder.features[:6](x)
        # # print(enc_1.size())
        # enc_2 = self.encoder.features[6:13](enc_1)
        # # print(enc_2.size())
        # enc_3 = self.encoder.features[13:26](enc_2)
        # # print(enc_3.size())
        # enc_4 = self.encoder.features[26:39](enc_3)
        # # print(enc_4.size())
        #
        # dec_1 = self.decoder[:12](enc_4) + enc_4
        # # print(dec_1.size())
        # dec_2 = self.decoder[12:25](dec_1) + enc_3
        # # print(dec_2.size())
        # dec_3 = self.decoder[25:](dec_2)
        # # print(dec_3.size())

        # # Arch 'B'
        # enc_1 = self.encoder.features[:6](x)
        # enc_2 = self.encoder.features[6:13](enc_1)
        # enc_3 = self.encoder.features[13:20](enc_2)
        # enc_4 = self.encoder.features[20:27](enc_3)
        #
        # dec_1 = self.decoder[:6](enc_4) + enc_4
        # dec_2 = self.decoder[6:13](dec_1) + enc_3
        # dec_3 = self.decoder[13:](dec_2)

        # Arch 'A'
        enc_1 = self.encoder.features[:3](x)
        print(enc_1.size())
        enc_2 = self.encoder.features[3:7](enc_1)
        print(enc_2.size())
        enc_3 = self.encoder.features[7:14](enc_2)
        print(enc_3.size())
        enc_4 = self.encoder.features[14:21](enc_3)
        print(enc_4.size())

        dec_1 = self.decoder[:6](enc_4) + enc_4
        print(dec_1.size())
        dec_2 = self.decoder[6:13](dec_1)
        print(dec_2.size())
        dec_3 = self.decoder[13:](dec_2)

        return dec_3


def show(img):
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


class VanillaAE(AE):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, input_height, enc_type='resnet18', first_conv=False, maxpool1=False, enc_out_dim=512,
                 kl_coeff=0.1, latent_dim=256, lr=1e-4, **kwargs):
        super(VanillaAE, self).__init__(input_height=input_height, enc_type=enc_type, first_conv=first_conv,
                                        maxpool1=maxpool1, enc_out_dim=enc_out_dim, kl_coeff=kl_coeff,
                                        latent_dim=latent_dim, lr=lr, **kwargs)

    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}


if __name__ == '__main__':
    # inp = torch.randn((1, 512, 243, 177))
    # out = make_layers(vgg_decoder_arch, batch_norm=True, input_size=(1945, 1422))
    # print(out(inp).size())

    from torchvision.utils import make_grid

    inp = torch.randn((1, 3, 1945, 1422))
    # inp = F.interpolate(inp, scale_factor=0.25)
    inp = F.interpolate(inp, size=(640, 480))
    print(inp.size())
    # encoder = vgg11_bn
    # decoder = make_layers(vgg_decoder_arch['A'], batch_norm=True)
    # # vgg = UnsupervisedTP(encoder, decoder)
    # vgg = VanillaAutoEncoder(encoder, decoder)
    # vgg.to('cuda')
    # print(vgg)
    # # xx = vgg(inp, torch.zeros((3, 1, h, w)).cuda())
    # xx = vgg(inp)
    # print(xx.size())
    # gg = make_grid(inp, nrow=3, padding=10)
    # show(gg)
    ae = VanillaAE(input_height=inp.size(2), enc_type='resnet50', enc_out_dim=2048, latent_dim=2048)
    print(ae)
    # ae = ae.from_pretrained(checkpoint_name='resnet50-imagenet')
    out = ae(inp)
    print(out.size())
