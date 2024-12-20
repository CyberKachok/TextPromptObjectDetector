import torch
from torch import nn


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()

        def create_block(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.GroupNorm(out_channels, out_channels),
                nn.ReLU(inplace=True)
            )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(768, 768, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(768, 768),
            nn.ReLU(inplace=True),
        )

        self.downsample_blocks = nn.ModuleList([
            nn.Sequential(
                create_block(768, 768, 3, 1, 0),
                create_block(768, 512, 4, 1, 0),
                create_block(512, 512, 3, 1, 0),
                create_block(512, 512, 3, 1, 0),
            ),
            nn.Sequential(
                create_block(768, 768, 3, 1, 0),
                create_block(768, 512, 3, 1, 0),
            ),
            nn.Sequential(
                create_block(768, 768, 3, 2, 1),
                create_block(768, 512, 3, 1, 1),
                create_block(512, 512, 3, 1, 1),
                create_block(512, 512, 3, 1, 0),
            ),
            nn.Sequential(
                create_block(768, 768, 3, 2, 0),
                create_block(768, 512, 3, 1, 1),
                create_block(512, 512, 3, 1, 0),
                create_block(512, 512, 3, 1, 0),
            ),
            nn.Sequential(
                create_block(768, 768, 3, 2, 1),
                create_block(768, 512, 3, 1, 0),
                create_block(512, 512, 3, 1, 0),
                create_block(512, 512, 3, 1, 0),
                create_block(512, 512, 3, 1, 0),
            ),
        ])

        self.clss = nn.ModuleList([
                                      create_block(512, 12, 3, 1, 1) for _ in range(4)
                                  ] + [create_block(512, 8, 3, 1, 1)])

        self.locs = nn.ModuleList([
                                      create_block(512, 24, 3, 1, 1) for _ in range(4)
                                  ] + [create_block(512, 16, 3, 1, 1)])

    def forward(self, x):
        x[0] = self.upsample(x[0])

        for i in range(5):
            x[i] = self.downsample_blocks[i](x[i])

        cls_out, loc_out = [], []

        for i in range(5):
            cls_out.append(self.clss[i](x[i]))
            loc_out.append(self.locs[i](x[i]))

        cls_out = torch.cat([c.permute(0, 2, 3, 1).contiguous().view(c.size(0), -1) for c in cls_out], dim=1)
        loc_out = torch.cat([l.permute(0, 2, 3, 1).contiguous().view(l.size(0), -1) for l in loc_out], dim=1)

        cls_out = cls_out.view(cls_out.size(0), -1, 2)
        loc_out = loc_out.view(loc_out.size(0), -1, 4)

        return loc_out, cls_out