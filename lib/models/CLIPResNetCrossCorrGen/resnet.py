from torch import nn
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ExtraBlock(nn.Module):
    def __init__(self):
        super(ExtraBlock, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.in_channels = 128
        self.out_channels = 128

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out


from torchvision.models import resnet18, ResNet18_Weights


class SSD_ResNet18(nn.Module):
    def __init__(self, num_bboxes_s, num_labels=3, pretrained=False):
        super(SSD_ResNet18, self).__init__()

        self.num_bboxes_s = num_bboxes_s
        self.num_labels = num_labels

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=1)
        # if pretrained:
        # 	pretrained_model = resnet18(pretrained=True)
        # 	pretrained_dict = pretrained_model.state_dict()
        # 	weights_before = self.layer2[0].conv1.weight
        # 	self.load_state_dict(pretrained_dict, strict=False)
        # 	weights_after = self.layer2[0].conv1.weight
        #
        # 	print( torch.norm(weights_before - weights_after).item())

        self.layer4 = self._make_layer(128, 2, stride=1)

        self.extra_layers = nn.Sequential(*[ExtraBlock() for _ in range(5)])

        self.conf_layers, self.loc_layers = self._make_conf_loc_layers()

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _make_conf_loc_layers(self):
        conf_layers, loc_layers = [], []

        for i in range(0, 5):
            conf_layer = nn.Conv2d(128, self.num_bboxes_s[i] * self.num_labels, kernel_size=3, padding=1)
            loc_layer = nn.Conv2d(128, self.num_bboxes_s[i] * 4, kernel_size=3, padding=1)
            conf_layers += [conf_layer]
            loc_layers += [loc_layer]

        conf_layers = nn.ModuleList(conf_layers)
        loc_layers = nn.ModuleList(loc_layers)

        return conf_layers, loc_layers

    def forward(self, x):
        loc_s, conf_s = [], []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        res = []

        res.append(x)
        for i in range(4):
            x = self.extra_layers[i](x)
            res.append(x)

        # #print(x.shape)
        # conf_s.append(self.conf_layers[0](x).permute(0, 2, 3, 1).contiguous())
        # loc_s.append(self.loc_layers[0](x).permute(0, 2, 3, 1).contiguous())
        # for i in range(5):
        #
        #     x = self.extra_layers[i](x)
        #     #print(x.shape)
        #
        #     conf_s.append(self.conf_layers[i+1](x).permute(0, 2, 3, 1).contiguous())
        #     loc_s.append(self.loc_layers[i+1](x).permute(0, 2, 3, 1).contiguous())
        #
        # conf_s = torch.cat([o.view(o.size(0), -1) for o in conf_s], 1)
        # loc_s  = torch.cat([o.view(o.size(0), -1) for o in loc_s ], 1)
        #
        # conf_s = conf_s.view(conf_s.size(0), -1, self.num_labels)
        # loc_s  = loc_s .view(loc_s .size(0), -1, 4              )

        return res
