import math
import time
import torch
import torch.nn as nn
from thop import profile
from model.z1 import *
from model.z2 import *


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MEEMEdgeBranch(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        channels = [16, 32, 64, 128]

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, 1, 1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        self.meem0 = MEEM(in_dim=channels[0], hidden_dim=channels[0] // 2, width=4)
        self.down1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, 2, 1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, 2, 1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, 2, 1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.init_conv(x)
        e0 = self.meem0(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        return e0, e1, e2, e3


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


class EdgeGuidedFusion(nn.Module):
    def __init__(self, feat_channels, edge_channels):
        super().__init__()
        self.edge_proj = nn.Sequential(
            nn.Conv2d(edge_channels, feat_channels, 1, bias=False),
            nn.BatchNorm2d(feat_channels)
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))  # feat权重
        self.beta = nn.Parameter(torch.tensor(0.3))  # feat*edge权重
        self.gamma = nn.Parameter(torch.tensor(0.2))  # edge权重
    def forward(self, feat, edge):
        edge_feat = self.edge_proj(edge)
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        beta = torch.clamp(self.beta, 0.0, 1.0)
        gamma = torch.clamp(self.gamma, 0.0, 1.0)

        total = alpha + beta + gamma
        alpha = alpha / total
        beta = beta / total
        gamma = gamma / total

        # 三路融合
        return alpha * feat + beta * (feat * edge_feat) + gamma * edge_feat


class MDAFNet(nn.Module):
    def __init__(self, input_channels, block=ResNet):
        super().__init__()
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)

        # 编码器
        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])

        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])
        self.edge_branch = MEEMEdgeBranch(in_channels=input_channels)
        self.edge_fusion_enc_3 = EdgeGuidedFusion(param_channels[3], 128)
        self.edge_fusion_enc_2 = EdgeGuidedFusion(param_channels[2], 64)
        self.edge_fusion_enc_1 = EdgeGuidedFusion(param_channels[1], 32)
        self.edge_fusion_enc_0 = EdgeGuidedFusion(param_channels[0], 16)
        self.hwfe0 = HWFE(param_channels[0], residual_weight=0.9)  # 90%原始
        self.hwfe1 = HWFE(param_channels[1], residual_weight=0.85)  # 85%原始
        self.hwfe2 = HWFE(param_channels[2], residual_weight=0.7)  # 70%原始
        self.hwfe3 = HWFE(param_channels[3], residual_weight=0.6)  # 60%原始

        # 解码器
        self.decoder_3 = self._make_layer(param_channels[3] + param_channels[4], param_channels[3], block,
                                          param_blocks[2])
        self.decoder_2 = self._make_layer(param_channels[2] + param_channels[3], param_channels[2], block,
                                          param_blocks[1])
        self.decoder_1 = self._make_layer(param_channels[1] + param_channels[2], param_channels[1], block,
                                          param_blocks[0])
        self.decoder_0 = self._make_layer(param_channels[0] + param_channels[1], param_channels[0], block)

        # 输出层
        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)
        self.final = nn.Conv2d(4, 1, 3, 1, 1)

    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x, warm_flag):
        # 边缘特征提取
        edge_0, edge_1, edge_2, edge_3 = self.edge_branch(x)

        # 编码
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))
        x_m = self.middle_layer(self.pool(x_e3))

        # 边缘融合
        x_e0 = self.edge_fusion_enc_0(x_e0, edge_0)
        x_e1 = self.edge_fusion_enc_1(x_e1, edge_1)
        x_e2 = self.edge_fusion_enc_2(x_e2, edge_2)
        x_e3 = self.edge_fusion_enc_3(x_e3, edge_3)

        # 分层特征增强
        x_e0 = self.hwfe0(x_e0)  #
        x_e1 = self.hwfe1(x_e1)  #
        x_e2 = self.hwfe2(x_e2)  # HWFE
        x_e3 = self.hwfe3(x_e3)  # HWFE

        # 解码
        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))

        if warm_flag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)
            output = self.final(torch.cat([mask0, self.up(mask1),
                                           self.up_4(mask2), self.up_8(mask3)], dim=1))
            return [mask0, mask1, mask2, mask3], output
        else:
            output = self.output_0(x_d0)
            return [], output

