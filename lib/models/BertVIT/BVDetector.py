import types
import torch
from torch import nn
import transformers
from transformers import AutoModel
from torchvision import models

from lib.models.BertVIT.bert import get_bert_model
from lib.models.BertVIT.vit import vision_transformer
from lib.models.BertVIT.head import Head

class BVPromptDetector(nn.Module):
    def __init__(self,):
        super().__init__()
        self.text_model = get_bert_model('cuda')

        self.combined_model = vision_transformer(
            image_size=320,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            weights=None,
            progress=False
        )
        combined_model_state_dict = torch.load("/home/ilya/code/VisNav/Training/PreTrainVIT/pretrained_timm.pth", weights_only=True)
        del combined_model_state_dict['heads.head.weight']
        del combined_model_state_dict['heads.head.bias']
        del combined_model_state_dict['pos_embed_z']
        del combined_model_state_dict['pos_embed_x']
        del combined_model_state_dict['encoder.pos_embedding']
        self.combined_model.load_state_dict(combined_model_state_dict, strict=False)
        #self.combined_model.initialize_weights()
        self.head = Head()

    def forward(self, text_tokens, image):
        #print(text_tokens)
        text_emb = self.text_model(**text_tokens)

        image_emb = self.combined_model(text_emb, image)

        return self.head(image_emb)

#return res
