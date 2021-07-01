import torch
from mmcv.cnn import build_norm_layer
from torch import nn


class SingleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size=3, stride=1, padding=1, norm_type='BN', norm_grad=True,
                 as_last_layer=False):
        super().__init__()
        norm_cfg = dict(type=norm_type, requires_grad=norm_grad)
        if as_last_layer:
            self.net = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                build_norm_layer(norm_cfg, out_ch)[-1], nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.net(x)


class DoubleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size=3, stride=1, padding=1, norm_type='BN', norm_grad=True,
                 as_last_layer=False):
        super().__init__()
        norm_cfg = dict(type=norm_type, requires_grad=norm_grad)
        if as_last_layer:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                build_norm_layer(norm_cfg, out_ch)[-1], nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                build_norm_layer(norm_cfg, out_ch)[-1], nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                build_norm_layer(norm_cfg, out_ch)[-1], nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int, use_conv_trans2d=True, bilinear: bool = False, channels_div_factor=1,
                 as_last_layer=False, use_double_conv=True, skip_double_conv=False):
        super().__init__()
        self.up = None
        self.skip_double_conv = skip_double_conv
        if use_conv_trans2d:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // channels_div_factor, kernel_size=2, stride=2)
        else:
            if bilinear:
                self.up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(in_ch, in_ch // channels_div_factor, kernel_size=1),
                )
            else:
                self.up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_ch, in_ch // channels_div_factor, kernel_size=1),
                )

        if not skip_double_conv:
            self.conv = DoubleConv(in_ch, out_ch, as_last_layer=as_last_layer) \
                if use_double_conv else SingleConv(in_ch, out_ch, as_last_layer=as_last_layer)

    def forward(self, x):
        x = self.up(x)
        if self.skip_double_conv:
            return x
        return self.conv(x)


class UpProject(nn.Module):  # loss goes to nan

    def __init__(self, in_ch: int, out_ch: int, use_conv_trans2d=True, bilinear: bool = False, channels_div_factor=1,
                 as_last_layer=False, use_double_conv=True):
        super(UpProject, self).__init__()

        self.conv1_1 = nn.Conv2d(in_ch, out_ch, 3)
        self.conv1_2 = nn.Conv2d(in_ch, out_ch, (2, 3))
        self.conv1_3 = nn.Conv2d(in_ch, out_ch, (3, 2))
        self.conv1_4 = nn.Conv2d(in_ch, out_ch, 2)

        self.conv2_1 = nn.Conv2d(in_ch, out_ch, 3)
        self.conv2_2 = nn.Conv2d(in_ch, out_ch, (2, 3))
        self.conv2_3 = nn.Conv2d(in_ch, out_ch, (3, 2))
        self.conv2_4 = nn.Conv2d(in_ch, out_ch, 2)

        self.bn1_1 = nn.BatchNorm2d(out_ch)
        self.bn1_2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        batch_size = x.shape[0]

        out1_1 = self.conv1_1(nn.functional.pad(x, (1, 1, 1, 1)))
        #out1_2 = self.conv1_2(nn.functional.pad(x, (1, 1, 0, 1)))#right interleaving padding
        out1_2 = self.conv1_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github
        #out1_3 = self.conv1_3(nn.functional.pad(x, (0, 1, 1, 1)))#right interleaving padding
        out1_3 = self.conv1_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github
        #out1_4 = self.conv1_4(nn.functional.pad(x, (0, 1, 0, 1)))#right interleaving padding
        out1_4 = self.conv1_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

        out2_1 = self.conv2_1(nn.functional.pad(x, (1, 1, 1, 1)))
        #out2_2 = self.conv2_2(nn.functional.pad(x, (1, 1, 0, 1)))#right interleaving padding
        out2_2 = self.conv2_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github
        #out2_3 = self.conv2_3(nn.functional.pad(x, (0, 1, 1, 1)))#right interleaving padding
        out2_3 = self.conv2_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github
        #out2_4 = self.conv2_4(nn.functional.pad(x, (0, 1, 0, 1)))#right interleaving padding
        out2_4 = self.conv2_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

        height = out1_1.size()[2]
        width = out1_1.size()[3]

        out1_1_2 = torch.stack((out1_1, out1_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out1_3_4 = torch.stack((out1_3, out1_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out1_1234 = torch.stack((out1_1_2, out1_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out2_1_2 = torch.stack((out2_1, out2_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out2_3_4 = torch.stack((out2_3, out2_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out2_1234 = torch.stack((out2_1_2, out2_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out1 = self.bn1_1(out1_1234)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn2(out1)

        out2 = self.bn1_2(out2_1234)

        out = out1 + out2
        out = self.relu(out)

        return out
