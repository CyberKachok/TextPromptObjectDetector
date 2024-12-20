import types
import open_clip

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = 100
        self.embed_dim = 1024
        self.projected_embed_dim = 128
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        self.ngf = 64

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
            nn.BatchNorm1d(num_features=self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.netG = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channels) x 64 x 64
        )

    def forward(self, embed_vector, z):
        projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
        latent_vector = torch.cat([projected_embed, z], 1)
        output = self.netG(latent_vector)

        return output


def forward(self, x: torch.Tensor, attn_mask=None):
    res = []
    for i, r in enumerate(self.resblocks):
        x = r(x, attn_mask=attn_mask)
        if i in (6, 7, 8, 9, 11):
            res.append(x)
    return res


class TensorTransformer(nn.Module):
    def __init__(self, feature_dim=128, depth=0):
        super(TensorTransformer, self).__init__()
        self.feature_dim = feature_dim
        self.col_embed = nn.Embedding(77, feature_dim)

        groupchannel = feature_dim
        self.convcls = nn.Sequential(  # TODO: try add CBAM modules
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groupchannel, feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groupchannel, feature_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=0),
            nn.GroupNorm(groupchannel, feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groupchannel, feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=0),
            nn.GroupNorm(groupchannel, feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groupchannel, feature_dim),
            nn.ReLU(inplace=True),

        )
        self.depth = depth
        for i in range(max(depth, 3)):
            self.add_module('downsample' + str(i),
                            nn.Sequential(
                                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=2),
                                nn.GroupNorm(groupchannel, feature_dim),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
                                nn.GroupNorm(groupchannel, feature_dim),
                                nn.ReLU(inplace=True),
                            )
                            )
        if depth == 4:
            self.add_module('downsample' + str(depth - 1),
                            nn.Sequential(
                                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
                                nn.GroupNorm(groupchannel, feature_dim),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
                                nn.GroupNorm(groupchannel, feature_dim),
                                nn.ReLU(inplace=True),
                            )
                            )

        self.gen = generator()
        model_state_dict = torch.load("/home/ilya/MEGA/Sem3/ML3/Generator2/flowers_generator.pth", weights_only=True)
        del model_state_dict['netG.0.weight']
        self.gen.load_state_dict(model_state_dict, strict=False)
        self.gen.netG[12] = nn.ConvTranspose2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        z = torch.randn(x.shape[0], 100, 1, 1).cuda()
        # TODO: cls token
        x = x.mean(dim=1)
        x = self.gen(x, z)
        x = self.convcls(x)

        for idx in range(self.depth):
            ds = getattr(self, 'downsample' + str(idx))
            x = ds(x)

        return x


def encode_text(self, text):
    # print('hi')
    cast_dtype = self.transformer.get_cast_dtype()

    x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

    x = x + self.positional_embedding.to(cast_dtype)
    x = self.transformer(x, attn_mask=None)

    for idx in range(5):
        # print(x[i].shape)
        tti = getattr(self, 'tti' + str(idx))
        # x[i] = self.ln_final(x[i])
        proj = getattr(self, 'text_projection' + str(idx))
        x[idx] = x[idx] @ proj
        x[idx] = tti(x[idx])

    return x


def get_text_model():
    model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
    del model.visual
    model.encode_text = types.MethodType(encode_text, model)
    model.transformer.forward = types.MethodType(forward, model.transformer)
    for i in range(5):
        model.add_module('tti' + str(i),
                         TensorTransformer(128, depth=i))

    model.text_projection0 = nn.Parameter(torch.rand(512, 1024))
    model.text_projection1 = nn.Parameter(torch.rand(512, 1024))
    model.text_projection2 = nn.Parameter(torch.rand(512, 1024))
    model.text_projection3 = nn.Parameter(torch.rand(512, 1024))
    model.text_projection4 = nn.Parameter(torch.rand(512, 1024))

    # model.positional_embedding = torch.nn.Parameter(torch.rand(225, 512))

    return model
