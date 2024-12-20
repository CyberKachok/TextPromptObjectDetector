import torch

from lib.models.CLIPResNetCrossCorrGen.rpn import MultiRPN
from lib.models.CLIPResNetCrossCorrGen.clip_text_encoder import get_text_model
from lib.models.CLIPResNetCrossCorrGen.resnet import SSD_ResNet18
from torch import nn


class CCPromptDetector(nn.Module):
    def __init__(self, custom_config):
        super(CCPromptDetector, self).__init__()
        self.text = get_text_model()
        self.visual = SSD_ResNet18(custom_config['num_priors'], custom_config['num_classes'])
        self.rpn_head = MultiRPN(5)

    def forward(self, data):
        template = data['template'].cuda()
        search = data['search'].cuda()
        xf = self.visual(search)
        zf = self.text.encode_text(template)

        cls, loc = self.rpn_head(zf, xf)

        conf_s, loc_s = [], []

        for conf, locf in zip(cls, loc):
            conf_s.append(conf.permute(0, 2, 3, 1).contiguous())
            loc_s.append(locf.permute(0, 2, 3, 1).contiguous())

        conf_s = torch.cat([o.view(o.size(0), -1) for o in conf_s], 1)
        loc_s = torch.cat([o.view(o.size(0), -1) for o in loc_s], 1)

        conf_s = conf_s.view(conf_s.size(0), -1, 2)
        loc_s = loc_s.view(loc_s.size(0), -1, 4)

        return loc_s, conf_s
