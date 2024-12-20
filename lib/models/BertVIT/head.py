import torch
from torch import nn


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(768, 768, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(768),
            nn.GroupNorm(768, 768),
            nn.ReLU(True),
        )

        self.downsample1 = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=0),
            nn.GroupNorm(768, 768),
            # nn.BatchNorm2d(768),
            nn.ReLU(True),
            nn.Conv2d(768, 512, kernel_size=4, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.GroupNorm(512, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.GroupNorm(512, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.GroupNorm(512, 512),
            nn.ReLU(True),
        )

        self.downsample2 = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(768),
            nn.GroupNorm(768, 768),
            nn.ReLU(True),
            nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.GroupNorm(512, 512),
            nn.ReLU(True),
        )

        self.downsample3 = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(768),
            nn.GroupNorm(768, 768),
            nn.ReLU(True),
            nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.GroupNorm(512, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.GroupNorm(512, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.GroupNorm(512, 512),
            nn.ReLU(True),
        )

        self.downsample4 = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=0),
            # nn.BatchNorm2d(768),
            nn.GroupNorm(768, 768),
            nn.ReLU(True),
            nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.GroupNorm(512, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.GroupNorm(512, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.GroupNorm(512, 512),
            nn.ReLU(True),
        )

        self.downsample5 = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(768),
            nn.GroupNorm(768, 768),
            nn.ReLU(True),
            nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.GroupNorm(512, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.GroupNorm(512, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.GroupNorm(512, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.GroupNorm(512, 512),
            nn.ReLU(True),
        )

        self.clss = nn.Sequential(*([nn.Conv2d(512, 12,  kernel_size=3, stride=1, padding=1) for _ in range(4)]
                                    + [nn.Conv2d(512, 8,  kernel_size=3, stride=1, padding=1)]))
        self.locs = nn.Sequential(*([nn.Conv2d(512, 24,  kernel_size=3, stride=1, padding=1) for _ in range(4)]
                                    + [nn.Conv2d(512, 16,  kernel_size=3, stride=1, padding=1)]))

    def forward(self, x):
        x[0] = self.upsample(x[0])
        x[0] = self.downsample1(x[0])
        x[1] = self.downsample2(x[1])
        x[2] = self.downsample3(x[2])
        x[3] = self.downsample4(x[3])
        x[4] = self.downsample5(x[4])

        cls, loc = [], []
        for i in range(5):
            cls.append(self.clss[i](x[i]))
            loc.append(self.locs[i](x[i]))

        conf_s, loc_s = [], []

        for conf, locf in zip(cls, loc):
            conf_s.append(conf.permute(0, 2, 3, 1).contiguous())
            loc_s.append(locf.permute(0, 2, 3, 1).contiguous())

        conf_s = torch.cat([o.view(o.size(0), -1) for o in conf_s], 1)
        loc_s = torch.cat([o.view(o.size(0), -1) for o in loc_s], 1)

        N_labels = 2
        conf_s = conf_s.view(conf_s.size(0), -1, N_labels)
        loc_s = loc_s.view(loc_s.size(0), -1, 4)
        # print(loc_s.shape)

        return loc_s, conf_s
