import torch.nn as nn
import torch
import math
from model.resnet_fsr import resnet18, resnet34, resnet50, resnet101


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
        channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x



class FSR_SHUFFLE_MODULE_X4(nn.Module):
    def __init__(self, in_planes, out_planes, feat_trim, num_labels, groups=1):
        super(FSR_SHUFFLE_MODULE_X4, self).__init__()
        self.feat_trim = feat_trim
        self.num_labels = num_labels
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups

        self.up_sample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.deconv1 = nn.ConvTranspose2d(in_planes, in_planes, kernel_size=4, stride=2, padding=1, groups=groups)

        self.up_sample2_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_sample2_2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.deconv2 = nn.ConvTranspose2d(out_planes, out_planes, kernel_size=4, stride=2, padding=1, groups=groups)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_all = nn.Linear(out_planes, num_labels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        upsampled_x = self.up_sample1(x)
        residual_x1 = self.conv1(x)
        residual_x1 = self.relu(residual_x1)
        residual_x1 = channel_shuffle(residual_x1, self.groups)
        residual_x1 = self.deconv1(residual_x1)
        residual_x1 = channel_shuffle(residual_x1, self.groups)
        x1 = upsampled_x + residual_x1

        upsampled_x1 = self.up_sample2_1(x1)
        if self.in_planes != self.out_planes:
            upsampled_x1 = self.up_sample2_2(upsampled_x1)
        residual_x2 = self.conv2(x1)
        residual_x2 = self.relu(residual_x2)
        residual_x2 = channel_shuffle(residual_x2, self.groups)
        residual_x2 = self.deconv2(residual_x2)
        residual_x2 = channel_shuffle(residual_x2, self.groups)
        x2 = upsampled_x1 + residual_x2

        # cls
        N, C, H, W = x2.shape
        x2 = x2[:, :, self.feat_trim[0]:H-self.feat_trim[1], self.feat_trim[0]:W-self.feat_trim[1]]
        y = self.avgpool(x2)
        y = y.view(y.size(0), -1)
        y = self.fc_all(y)

        feature_map = x2.detach().clone()
        fc_weights = self.fc_all.weight.view(
            1, self.num_labels, feature_map.shape[1], 1, 1)  # 1 * L * C * 1 * 1
        feature = feature_map.unsqueeze(1)  # N * 1 * C * H * W
        cams = (feature * fc_weights).sum(2)  # N * L * H * W

        return y, x2, cams


class FSR_SHUFFLE_X4(nn.Module):
    def __init__(self, arch, in_planes, out_planes, feat_trim, num_labels, groups=1):
        super(FSR_SHUFFLE_X4, self).__init__()

        if arch == 'r101':
            self.base_model = resnet101(pretrained=True, num_labels=num_labels)
        elif arch == 'r50':
            self.base_model = resnet50(pretrained=True, num_labels=num_labels)
        elif arch == 'r34':
            self.base_model = resnet34(pretrained=True, num_labels=num_labels)
        elif arch == 'r18':
            self.base_model = resnet18(pretrained=True, num_labels=num_labels)
        self.fsr_model = FSR_SHUFFLE_MODULE_X4(in_planes, out_planes, feat_trim, num_labels, groups)

    def forward(self, x):
        base_cls, base_m_feat, base_f_feat = self.base_model(x)
        fsr_cls, fsr_feat, fsr_cam = self.fsr_model(base_f_feat)
        return base_cls, base_m_feat, base_f_feat, fsr_cls, fsr_feat, fsr_cam

if __name__ == "__main__":
    from thop import profile
    fsr_kd_model = FSR_SHUFFLE_X4(arch='r18', in_planes=512, out_planes=512, feat_trim=[1,1], num_labels=80, groups=64)
    input = torch.randn(1, 3, 112, 112)
    flops, params = profile(fsr_kd_model, inputs=(input, ))
    print('flops: {}\tparams: {}'.format(flops, params))
    # base_cls, base_m_feat, base_f_feature, fsr_cls, fsr_feat, fsr_cam = fsr_kd_model(input)

