# Custom modules for PCG-YOLO (CACG/CGLU-style block).
# Designed to be copied into Ultralytics via patch_ultralytics.py.

from __future__ import annotations
import math
import torch
import torch.nn as nn


def autopad(k: int, p: int | None = None) -> int:
    return k // 2 if p is None else p


class Conv(nn.Module):
    """Conv2d + BN + SiLU (Ultralytics-like)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depthwise conv."""
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__(c1, c2, k=k, s=s, g=math.gcd(c1, c2), act=act)


class ConvGLU(nn.Module):
    """Convolutional GLU: y = A(x) * sigmoid(B(x))"""
    def __init__(self, c, k=1):
        super().__init__()
        self.proj = Conv(c, 2 * c, k=k, s=1, act=False)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=1)
        return a * torch.sigmoid(b)


class CASA(nn.Module):
    """Convolutional additive attention (lightweight approximation)."""
    def __init__(self, c, k=3):
        super().__init__()
        self.qkv = Conv(c, 3 * c, k=1, s=1, act=False)
        self.dw = DWConv(c, c, k=k, s=1, act=False)
        self.proj = Conv(c, c, k=1, s=1, act=False)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        a = self.dw(q + k)
        a = torch.sigmoid(a)
        return self.proj(v * a)


class MLP(nn.Module):
    """1x1 conv MLP."""
    def __init__(self, c, mlp_ratio=2.0):
        super().__init__()
        hidden = int(c * mlp_ratio)
        self.fc1 = Conv(c, hidden, k=1, s=1, act=True)
        self.fc2 = Conv(hidden, c, k=1, s=1, act=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class CACGBlock(nn.Module):
    """CASA + ConvGLU + MLP with residual."""
    def __init__(self, c, mlp_ratio=2.0):
        super().__init__()
        self.attn = CASA(c)
        self.glu = ConvGLU(c)
        self.mlp = MLP(c, mlp_ratio=mlp_ratio)

    def forward(self, x):
        y = x + self.attn(x)
        y = y + self.glu(y)
        y = y + self.mlp(y)
        return y


class C3CACG(nn.Module):
    """CSP-like block using CACGBlock (drop-in for C3/C3k2)."""
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[CACGBlock(c_, mlp_ratio=2.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
