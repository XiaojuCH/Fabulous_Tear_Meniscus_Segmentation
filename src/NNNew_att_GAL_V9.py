import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------
# 🚀 预算消耗器 1：现代 ConvNeXt 风格瓶颈层 (提供庞大的特征提纯容量)
# -----------------------------------------------------------------
class ModernBottleneck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 7x7 大核深度卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        # 1x1 卷积放大 4 倍通道 (疯狂消耗预算，提升容量)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=False)
        self.act = nn.GELU()
        # 1x1 卷积降维
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + shortcut

# -----------------------------------------------------------------
# 🚀 预算消耗器 2：坐标注意力机制 (精确分离水平泪河与圆形反光)
# -----------------------------------------------------------------
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=8):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.GELU()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        return identity * a_h * a_w

# -----------------------------------------------------------------
# 基础 Strip 卷积层
# -----------------------------------------------------------------
class DWConv(nn.Module):
    def __init__(self, dim, kernel_size, padding, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))

# -----------------------------------------------------------------
# 终极版 GAL Adapter (V9)
# -----------------------------------------------------------------
class GAL_Adapter(nn.Module):
    def __init__(self, in_channels, kernel_size_large=23, kernel_size_mid=15, kernel_size_small=7, reduction=16):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        # 🚀 替换原有的单薄 pre_orient，接入双层 ConvNeXt 引擎
        self.pre_orient = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            ModernBottleneck(in_channels),
            ModernBottleneck(in_channels) # 堆叠两层，彻底榨干 0.5M 预算
        )

        pad_l = (kernel_size_large - 1) // 2
        pad_m = (kernel_size_mid - 1) // 2
        pad_s = (kernel_size_small - 1) // 2
        
        self.strip_h_large = DWConv(in_channels, (kernel_size_large, 1), (pad_l, 0))
        self.strip_w_large = DWConv(in_channels, (1, kernel_size_large), (0, pad_l))
        
        self.strip_h_mid = DWConv(in_channels, (kernel_size_mid, 1), (pad_m, 0))
        self.strip_w_mid = DWConv(in_channels, (1, kernel_size_mid), (0, pad_m))
        
        self.strip_h_small = DWConv(in_channels, (kernel_size_small, 1), (pad_s, 0))
        self.strip_w_small = DWConv(in_channels, (1, kernel_size_small), (0, pad_s))

        self.local_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.local_5x5 = DWConv(in_channels, 3, padding=2, dilation=2)

        # 8 分支像素级门控 (坚守最优信息瓶颈 reduction=16)
        num_branches = 8
        mid_channels = max(in_channels * num_branches // reduction, 16)
        
        self.branch_weight = nn.Sequential(
            nn.Conv2d(in_channels * num_branches, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels * num_branches, kernel_size=1, bias=False)
        )

        # 🚀 接入坐标注意力机制 (过滤融合后的特征)
        self.coord_att = CoordAtt(in_channels)

        # CCSM (恢复 V7 的 4C 大容量)
        style_hidden = in_channels * 4  
        self.style_fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, style_hidden, 1), 
            nn.GELU(),
            nn.Conv2d(style_hidden, in_channels * 2, 1)  
        )

        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        shortcut = x
        x = self.proj_in(x)

        x_oriented = self.pre_orient(x)

        # 8路并发
        lh = self.strip_h_large(x_oriented)
        lw = self.strip_w_large(x_oriented)
        mh = self.strip_h_mid(x_oriented)
        mw = self.strip_w_mid(x_oriented)
        sh = self.strip_h_small(x_oriented)
        sw = self.strip_w_small(x_oriented)
        loc3 = self.local_3x3(x)
        loc5 = self.local_5x5(x)

        branches = [lh, lw, mh, mw, sh, sw, loc3, loc5]

        # 空间融合
        cat_feat = torch.cat(branches, dim=1)
        weight = self.branch_weight(cat_feat)

        B, C8, H, W = weight.shape
        C = C8 // 8
        weight = weight.view(B, 8, C, H, W)
        weight = F.softmax(weight, dim=1) 

        stacked = torch.stack(branches, dim=1)
        out = (weight * stacked).sum(dim=1) 

        # 🚀 坐标注意力提纯 (强化水平拓扑，削弱环形伪影)
        out = self.coord_att(out)

        # 全局强度调制
        b, c, h, w = out.shape
        out_flat = out.view(b, c, -1)
        
        feat_mean = out_flat.mean(dim=2, keepdim=True).unsqueeze(-1) 
        var = out_flat.var(dim=2, keepdim=True, unbiased=False)
        feat_std = torch.sqrt(var + 1e-5).unsqueeze(-1) 
        
        style_feat = torch.cat([feat_mean, feat_std], dim=1)
        
        gamma_beta = self.style_fc(style_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = 2.0 * torch.sigmoid(gamma) # 恢复原版
        
        out = gamma * out + beta
        out = self.proj_out(out)

        return shortcut + out