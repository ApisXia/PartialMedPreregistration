import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple_ConvMODEL(nn.Module):
    def __init__(self):
        super(Simple_ConvMODEL, self).__init__()
        self.ra_conv_in = nn.Conv3d(1, 64, 3, padding=1)  # out: 128
        self.ra_conv_0 = nn.Conv3d(64, 128, 3, padding=1)  # out: 64
        self.ra_conv_0_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 64
        self.ra_conv_1 = nn.Conv3d(128, 256, 3, padding=1)  # out: 32
        self.ra_conv_1_1 = nn.Conv3d(256, 256, 3, padding=1)  # out: 32

        self.ra_conv_in_bn = nn.BatchNorm3d(64)
        self.ra_conv0_1_bn = nn.BatchNorm3d(128)
        self.ra_conv1_1_bn = nn.BatchNorm3d(256)

        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.actvn = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(2)

        self.fc_1 = nn.Linear(2048, 256)
        self.dropout = nn.Dropout()
        self.fc_2 = nn.Linear(256, 3)

    def forward(self, ra):
        # 3DRA net part
        ra = torch.unsqueeze(ra, dim=1)
        ra_net = self.actvn(self.ra_conv_in_bn(self.ra_conv_in(ra)))
        ra_net = self.maxpool(ra_net)

        ra_net = self.actvn(self.ra_conv_0(ra_net))
        ra_net = self.actvn(self.ra_conv0_1_bn(self.ra_conv_0_1(ra_net)))
        ra_net = self.maxpool(ra_net)

        ra_net = self.actvn(self.ra_conv_1(ra_net))
        ra_net = self.actvn(self.ra_conv1_1_bn(self.ra_conv_1_1(ra_net)))

        ra_net = self.maxpool(ra_net)
        ra_net = self.flatten(ra_net)

        out = self.fc_1(ra_net)
        out = self.actvn(out)
        out = self.dropout(out)
        out = self.fc_2(out)

        return out


if __name__ == '__main__':
    model = Simple_ConvMODEL()
    model = model.cuda()
    ct_sample = torch.randn([1, 128, 128, 128]).cuda()
    ra_sample = torch.randn([1, 16, 16, 16]).cuda()
    out = model(ra_sample)
    c = 1