from typing import Tuple, Optional, Callable

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset

from base import Base
from resnext import ResNeXt101
from attention.multi_scale_attention import PAM_CAM_Layer, SemanticModule, MultiConv


# https://github.com/sinAshish/Multi-Scale-Attention/blob/96d76f2794ee1b16e847f38a16d397d137c774f6/src/models/my_stacked_danet.py#L19
class MSANet(Base):
    def __init__(self, config: DictConfig, train_dataset: Dataset, val_dataset: Dataset,
                 desired_output_shape: Tuple[int, int] = None, loss_function: nn.Module = None,
                 collate_fn: Optional[Callable] = None):
        super(MSANet, self).__init__(
            config=config, train_dataset=train_dataset, val_dataset=val_dataset,
            desired_output_shape=desired_output_shape, loss_function=loss_function, collate_fn=collate_fn
        )

        self.resnext = ResNeXt101()

        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        inter_channels = 64
        out_channels = 64

        self.conv6_1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))

        self.conv7_1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))

        self.conv8_1 = nn.Conv2d(64, 64, 1)
        self.conv8_2 = nn.Conv2d(64, 64, 1)
        self.conv8_3 = nn.Conv2d(64, 64, 1)
        self.conv8_4 = nn.Conv2d(64, 64, 1)
        self.conv8_11 = nn.Conv2d(64, 64, 1)
        self.conv8_12 = nn.Conv2d(64, 64, 1)
        self.conv8_13 = nn.Conv2d(64, 64, 1)
        self.conv8_14 = nn.Conv2d(64, 64, 1)

        self.softmax_1 = nn.Softmax(dim=-1)

        self.pam_attention_1_1 = PAM_CAM_Layer(64, True)
        self.cam_attention_1_1 = PAM_CAM_Layer(64, False)
        self.semanticModule_1_1 = SemanticModule(128)

        self.conv_sem_1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Dual Attention mechanism
        self.pam_attention_1_2 = PAM_CAM_Layer(64)
        self.cam_attention_1_2 = PAM_CAM_Layer(64, False)
        self.pam_attention_1_3 = PAM_CAM_Layer(64)
        self.cam_attention_1_3 = PAM_CAM_Layer(64, False)
        self.pam_attention_1_4 = PAM_CAM_Layer(64)
        self.cam_attention_1_4 = PAM_CAM_Layer(64, False)

        self.pam_attention_2_1 = PAM_CAM_Layer(64)
        self.cam_attention_2_1 = PAM_CAM_Layer(64, False)
        self.semanticModule_2_1 = SemanticModule(128)

        self.conv_sem_2_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.pam_attention_2_2 = PAM_CAM_Layer(64)
        self.cam_attention_2_2 = PAM_CAM_Layer(64, False)
        self.pam_attention_2_3 = PAM_CAM_Layer(64)
        self.cam_attention_2_3 = PAM_CAM_Layer(64, False)
        self.pam_attention_2_4 = PAM_CAM_Layer(64)
        self.cam_attention_2_4 = PAM_CAM_Layer(64, False)

        self.fuse1 = MultiConv(256, 64, False)

        self.attention4 = MultiConv(128, 64)
        self.attention3 = MultiConv(128, 64)
        self.attention2 = MultiConv(128, 64)
        self.attention1 = MultiConv(128, 64)

        self.refine4 = MultiConv(128, 64, False)
        self.refine3 = MultiConv(128, 64, False)
        self.refine2 = MultiConv(128, 64, False)
        self.refine1 = MultiConv(128, 64, False)

        self.predict4 = nn.Conv2d(64, 1, kernel_size=1)
        self.predict3 = nn.Conv2d(64, 1, kernel_size=1)
        self.predict2 = nn.Conv2d(64, 1, kernel_size=1)
        self.predict1 = nn.Conv2d(64, 1, kernel_size=1)

        self.predict4_2 = nn.Conv2d(64, 1, kernel_size=1)
        self.predict3_2 = nn.Conv2d(64, 1, kernel_size=1)
        self.predict2_2 = nn.Conv2d(64, 1, kernel_size=1)
        self.predict1_2 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        layer0 = self.resnext.layer0(x)
        layer1 = self.resnext.layer1(layer0)
        layer2 = self.resnext.layer2(layer1)
        layer3 = self.resnext.layer3(layer2)
        layer4 = self.resnext.layer4(layer3)

        down4 = F.upsample(self.down4(layer4), size=layer1.size()[2:], mode='bilinear')
        down3 = F.upsample(self.down3(layer3), size=layer1.size()[2:], mode='bilinear')
        down2 = F.upsample(self.down2(layer2), size=layer1.size()[2:], mode='bilinear')
        down1 = self.down1(layer1)

        predict4 = self.predict4(down4)
        predict3 = self.predict3(down3)
        predict2 = self.predict2(down2)
        predict1 = self.predict1(down1)

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))

        semVector_1_1, semanticModule_1_1 = self.semanticModule_1_1(torch.cat((down4, fuse1), 1))

        attn_pam4 = self.pam_attention_1_4(torch.cat((down4, fuse1), 1))
        attn_cam4 = self.cam_attention_1_4(torch.cat((down4, fuse1), 1))

        attention1_4 = self.conv8_1((attn_cam4 + attn_pam4) * self.conv_sem_1_1(semanticModule_1_1))

        semVector_1_2, semanticModule_1_2 = self.semanticModule_1_1(torch.cat((down3, fuse1), 1))
        attn_pam3 = self.pam_attention_1_3(torch.cat((down3, fuse1), 1))
        attn_cam3 = self.cam_attention_1_3(torch.cat((down3, fuse1), 1))
        attention1_3 = self.conv8_2((attn_cam3 + attn_pam3) * self.conv_sem_1_2(semanticModule_1_2))

        semVector_1_3, semanticModule_1_3 = self.semanticModule_1_1(torch.cat((down2, fuse1), 1))
        attn_pam2 = self.pam_attention_1_2(torch.cat((down2, fuse1), 1))
        attn_cam2 = self.cam_attention_1_2(torch.cat((down2, fuse1), 1))
        attention1_2 = self.conv8_3((attn_cam2 + attn_pam2) * self.conv_sem_1_3(semanticModule_1_3))

        semVector_1_4, semanticModule_1_4 = self.semanticModule_1_1(torch.cat((down1, fuse1), 1))
        attn_pam1 = self.pam_attention_1_1(torch.cat((down1, fuse1), 1))
        attn_cam1 = self.cam_attention_1_1(torch.cat((down1, fuse1), 1))
        attention1_1 = self.conv8_4((attn_cam1 + attn_pam1) * self.conv_sem_1_4(semanticModule_1_4))

        # new design with stacked attention

        semVector_2_1, semanticModule_2_1 = self.semanticModule_2_1(torch.cat((down4, attention1_4 * fuse1), 1))

        refine4_1 = self.pam_attention_2_4(torch.cat((down4, attention1_4 * fuse1), 1))
        refine4_2 = self.cam_attention_2_4(torch.cat((down4, attention1_4 * fuse1), 1))
        refine4 = self.conv8_11((refine4_1 + refine4_2) * self.conv_sem_2_1(semanticModule_2_1))

        semVector_2_2, semanticModule_2_2 = self.semanticModule_2_1(torch.cat((down3, attention1_3 * fuse1), 1))
        refine3_1 = self.pam_attention_2_3(torch.cat((down3, attention1_3 * fuse1), 1))
        refine3_2 = self.cam_attention_2_3(torch.cat((down3, attention1_3 * fuse1), 1))
        refine3 = self.conv8_12((refine3_1 + refine3_2) * self.conv_sem_2_2(semanticModule_2_2))

        semVector_2_3, semanticModule_2_3 = self.semanticModule_2_1(torch.cat((down2, attention1_2 * fuse1), 1))
        refine2_1 = self.pam_attention_2_2(torch.cat((down2, attention1_2 * fuse1), 1))
        refine2_2 = self.cam_attention_2_2(torch.cat((down2, attention1_2 * fuse1), 1))
        refine2 = self.conv8_13((refine2_1 + refine2_2) * self.conv_sem_2_3(semanticModule_2_3))

        semVector_2_4, semanticModule_2_4 = self.semanticModule_2_1(torch.cat((down1, attention1_1 * fuse1), 1))
        refine1_1 = self.pam_attention_2_1(torch.cat((down1, attention1_1 * fuse1), 1))
        refine1_2 = self.cam_attention_2_1(torch.cat((down1, attention1_1 * fuse1), 1))

        refine1 = self.conv8_14((refine1_1 + refine1_2) * self.conv_sem_2_4(semanticModule_2_4))

        predict4_2 = self.predict4_2(refine4)
        predict3_2 = self.predict3_2(refine3)
        predict2_2 = self.predict2_2(refine2)
        predict1_2 = self.predict1_2(refine1)

        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear')
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear')
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear')
        predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear')

        predict1_2 = F.upsample(predict1_2, size=x.size()[2:], mode='bilinear')
        predict2_2 = F.upsample(predict2_2, size=x.size()[2:], mode='bilinear')
        predict3_2 = F.upsample(predict3_2, size=x.size()[2:], mode='bilinear')
        predict4_2 = F.upsample(predict4_2, size=x.size()[2:], mode='bilinear')

        if self.training:
            return semVector_1_1, \
                   semVector_2_1, \
                   semVector_1_2, \
                   semVector_2_2, \
                   semVector_1_3, \
                   semVector_2_3, \
                   semVector_1_4, \
                   semVector_2_4, \
                   torch.cat((down1, fuse1), 1), \
                   torch.cat((down2, fuse1), 1), \
                   torch.cat((down3, fuse1), 1), \
                   torch.cat((down4, fuse1), 1), \
                   torch.cat((down1, attention1_1 * fuse1), 1), \
                   torch.cat((down2, attention1_2 * fuse1), 1), \
                   torch.cat((down3, attention1_3 * fuse1), 1), \
                   torch.cat((down4, attention1_4 * fuse1), 1), \
                   semanticModule_1_4, \
                   semanticModule_1_3, \
                   semanticModule_1_2, \
                   semanticModule_1_1, \
                   semanticModule_2_4, \
                   semanticModule_2_3, \
                   semanticModule_2_2, \
                   semanticModule_2_1, \
                   predict1, \
                   predict2, \
                   predict3, \
                   predict4, \
                   predict1_2, \
                   predict2_2, \
                   predict3_2, \
                   predict4_2
        else:
            return (predict1_2 + predict2_2 + predict3_2 + predict4_2) / 4


if __name__ == '__main__':
    inp = torch.randn((1, 3, 480, 240))
    m = MSANet({}, None, None)
    o = m(inp)
    print()
