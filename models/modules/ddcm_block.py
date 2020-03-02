import torch
import torch.nn as nn


class ddcmBlock(nn.Module):
    def __init__(self, in_dim, out_dim, rates, kernel=3, bias=False, extend_dim=False):
        super(ddcmBlock, self).__init__()
        self.features = []
        self.num = len(rates)
        self.in_dim = in_dim
        self.out_dim = out_dim

        if self.num > 0:
            if extend_dim:
                self.out_dim = out_dim * self.num
            for idx, rate in enumerate(rates):
                self.features.append(nn.Sequential(
                    nn.Conv2d(self.in_dim + idx * out_dim,
                              out_dim,
                              kernel_size=kernel, dilation=rate,
                              padding=rate * (kernel - 1) // 2, bias=bias),
                    nn.PReLU(),
                    nn.BatchNorm2d(out_dim))
                )

            self.features = nn.ModuleList(self.features)

        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(self.in_dim + out_dim * self.num,
                      self.out_dim, kernel_size=1, bias=bias),
            nn.PReLU(),
            nn.BatchNorm2d(self.out_dim),
        )

    def forward(self, x):
        for f in self.features:
            x = torch.cat([f(x), x], 1)
        x = self.conv1x1_out(x)
        return x
