from typing import Tuple

import torch
from torch import nn
from torch.nn import MSELoss
from torch.nn.functional import interpolate

Pool = nn.MaxPool2d


def batch_norm(x):
    return nn.BatchNorm2d(x.size()[1])(x)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim,
                              kernel_size, stride,
                              padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n - 1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)

        try:
            out = up1 + up2
        except RuntimeError:
            # for non-square images
            up1 = interpolate(up1, size=(x.shape[-2], x.shape[-1]), mode='bilinear')
            up2 = interpolate(up2, size=(x.shape[-2], x.shape[-1]), mode='bilinear')
            out = up1 + up2
        return out


class UnFlatten(nn.Module):
    def __init__(self):
        super(UnFlatten, self).__init__()

    def forward(self, inp):
        return inp.view(-1, 256, 4, 4)


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):
    def __init__(self, num_stack, input_channels, num_classes, loss_fn, bn=False, increase=0,
                 desired_output_shape: Tuple[int, int] = None, **kwargs):
        super(PoseNet, self).__init__()

        self.num_stack = num_stack
        self.pre = nn.Sequential(
            Conv(input_channels, 64, 3, 1, bn=True, relu=True),  # originally 3, 64, 7, 2
            Residual(64, 128),
            # Pool(2, 2),
            Residual(128, 128),
            Residual(128, input_channels)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, input_channels, bn, increase),
            ) for _ in range(num_stack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(input_channels, input_channels),
                Conv(input_channels, input_channels, 1, bn=True, relu=True)
            ) for _ in range(num_stack)])

        self.outs = nn.ModuleList([Conv(input_channels, num_classes, 1, relu=False, bn=False) for i in range(num_stack)])
        self.merge_features = nn.ModuleList([Merge(input_channels, input_channels) for i in range(num_stack - 1)])
        self.merge_preds = nn.ModuleList([Merge(num_classes, input_channels) for i in range(num_stack - 1)])
        self.post_preds = nn.Sequential(
            Conv(num_classes, num_classes, 3, 1, bn=False, relu=False)
        )
        self.post_feature = nn.Sequential(
            Conv(input_channels, input_channels, 3, 1, bn=False, relu=False)
        )
        self.post_input = nn.Sequential(
            Conv(input_channels, input_channels, 3, 1, bn=False, relu=False)
        )

        self.num_stack = num_stack
        self.desired_output_shape = desired_output_shape

        self.loss_fn = loss_fn

    def forward(self, imgs):
        x = self.pre(imgs)
        if self.desired_output_shape is not None:
            x = interpolate(x, size=self.desired_output_shape)
            x = self.post_feature(x)

        combined_hm_preds = []
        for i in range(self.num_stack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)

            if self.desired_output_shape is not None:
                feature = interpolate(feature, size=self.desired_output_shape)
                feature = self.post_feature(feature)

                preds = interpolate(preds, size=self.desired_output_shape)
                preds = self.post_preds(preds)

            combined_hm_preds.append(preds)

            if i < self.num_stack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        return torch.stack(combined_hm_preds, dim=0)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.num_stack):
            # combined_loss.append(self.loss(combined_hm_preds[0][:, i], heatmaps))
            combined_loss.append(self.loss_fn(combined_hm_preds[i], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=0)

        return combined_loss


if __name__ == '__main__':
    inp = torch.randn((2, 3, 490, 320))
    # target = torch.randn((2, 1, 490, 320))
    target = torch.randn((2, 1, 190, 160))
    net = PoseNet(num_stack=3, input_channels=3, num_classes=1, loss_fn=MSELoss(), desired_output_shape=(190, 160))
    o = net(inp)
    loss = net.calc_loss(o, target)
    print()
