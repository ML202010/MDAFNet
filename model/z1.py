import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

import torch
import torch.nn as nn
import torch.nn.functional as F

class SSA(nn.Module):
    def __init__(self, dim, group=1, kernel=7) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att_unit(dim, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att_unit(dim, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


class spatial_strip_att_unit(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True) -> None:
        super().__init__()
        self.k = kernel
        pad = kernel // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel // 2, 1) if H else (1, kernel // 2)

        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group * kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()

    def forward(self, x):
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c // self.group, self.k, h * w)

        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1 // self.k, self.k, p * q).unsqueeze(2)
        filter = self.filter_act(filter)

        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out


class FrequencyStripAttention(nn.Module):
    def __init__(self, k, kernel=7) -> None:
        super().__init__()
        self.channel = k
        self.vert_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_high = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_high = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_pool = nn.AvgPool2d(kernel_size=(kernel, 1), stride=1)
        self.hori_pool = nn.AvgPool2d(kernel_size=(1, kernel), stride=1)
        pad_size = kernel // 2
        self.pad_vert = nn.ReflectionPad2d((0, 0, pad_size, pad_size))
        self.pad_hori = nn.ReflectionPad2d((pad_size, pad_size, 0, 0))
        self.gamma = nn.Parameter(torch.zeros(k, 1, 1))
        self.beta = nn.Parameter(torch.ones(k, 1, 1))

    def forward(self, x):
        hori_l = self.hori_pool(self.pad_hori(x))
        hori_h = x - hori_l
        hori_out = self.hori_low * hori_l + (self.hori_high + 1.) * hori_h
        vert_l = self.vert_pool(self.pad_vert(hori_out))
        vert_h = hori_out - vert_l
        vert_out = self.vert_low * vert_l + (self.vert_high + 1.) * vert_h
        return x * self.beta + vert_out * self.gamma


# ========== 快速版DWT/IDWT（GPU加速）==========
class FastDWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')

    def forward(self, x):
        """
        输入: x [B, C, H, W]
        输出: ll, lh, hl, hh 各为 [B, C, H/2, W/2]
        """
        Yl, Yh = self.dwt(x)
        # Yl: [B, C, H/2, W/2] - 低频
        # Yh: list of [B, C, 3, H/2, W/2] - 高频

        ll = Yl
        lh = Yh[0][:, :, 0, :, :]  # 第0个方向
        hl = Yh[0][:, :, 1, :, :]  # 第1个方向
        hh = Yh[0][:, :, 2, :, :]  # 第2个方向

        return ll, lh, hl, hh


class FastIDWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.idwt = DWTInverse(wave='haar', mode='zero')

    def forward(self, ll, lh, hl, hh):
        """
        输入: ll, lh, hl, hh 各为 [B, C, H, W]
        输出: x [B, C, H*2, W*2]
        """
        # 组装成DWTInverse需要的格式
        Yh = [torch.stack([lh, hl, hh], dim=2)]
        # Yh: list of [B, C, 3, H, W]

        x = self.idwt((ll, Yh))
        return x
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class MKP(nn.Module):
    def __init__(self, dim, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim, dim, 3,
            1, 1, groups=dim
        )
        self.conv2 = Conv(dim, dim, k=1, s=1, )
        self.conv3 = nn.Conv2d(
            dim, dim, 5,
            1, 2, groups=dim
        )
        self.conv4 = Conv(dim, dim, 1, 1)
        self.conv5 = nn.Conv2d(
            dim, dim, 7,
            1, 3, groups=dim
        )
        self.conv6 = Conv(dim, dim, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = x6 + x
        return x7


class HWFE(nn.Module):
    def __init__(self, channels, residual_weight=0.8):
        super(HWFE, self).__init__()
        self.fast_dwt = FastDWT()
        self.fast_idwt = FastIDWT()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.basic_block = MKP(dim=channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.BatchNorm2d(channels * 4),
            nn.GELU()
        )
        self.fsa = FrequencyStripAttention(k=channels, kernel=7)
        

        self.alpha = nn.Parameter(torch.tensor(residual_weight))  # 原始特征权重
        self.beta = nn.Parameter(torch.tensor(1.0 - residual_weight))  # HWFE特征权重

    def forward(self, x):
        f = x  # 原始特征
        
        ll, lh, hl, hh = self.fast_dwt(x)
        x = torch.cat([ll, lh, hl, hh], dim=1)
        x = self.conv1(x)
        x = self.basic_block(x)
        x = self.conv2(x)
        
        B, C4, H, W = x.shape
        C = C4 // 4
        ll = x[:, :C]
        lh = x[:, C:2 * C]
        hl = x[:, 2 * C:3 * C]
        hh = x[:, 3 * C:]
        
        out = self.fast_idwt(ll, lh, hl, hh)
        out = self.fsa(out)

        return torch.clamp(self.alpha, 0, 1) * f + torch.clamp(self.beta, 0, 1) * out

