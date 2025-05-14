
from torchvision.models.resnet import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, reduce
import pytorch_lightning as pl
from .pos_enc import ImageRotaryEmbed, ImgPosEnc

def _forward_reimpl(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x1 = self.layer1(x)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)

    return (x2,x3,x4)

class Encoder(pl.LightningModule):
    def __init__(self, d_model: int):
        super().__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.model.forward=_forward_reimpl.__get__(self.model)

        self.feature_proj2 = nn.Sequential(
            nn.Conv2d(256, d_model, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.feature_proj3 = nn.Sequential(
            nn.Conv2d(512, d_model, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)

    def forward(self, img, img_mask):
        img_mask1 = img_mask[:, 0::2, 0::2]
        img_mask2 = img_mask1[:, 0::2, 0::2]
        img_mask3 = img_mask2[:, 0::2, 0::2]
        img_mask4 = img_mask3[:, 0::2, 0::2]
        img_mask5 = img_mask4[:, 0::2, 0::2]

        f2,f3,f4 = self.model(img)

        f3 = self.feature_proj2(f3)
        f3 = rearrange(f3, "b d h w -> b h w d")
        f3 = self.norm2(f3)
        f3 = self.pos_enc_2d(f3, img_mask4)

        f4 = self.feature_proj3(f4)
        f4 = rearrange(f4, "b d h w -> b h w d")
        f4 = self.norm3(f4)
        f4 = self.pos_enc_2d(f4, img_mask5)

        shapes=[img_mask4.shape, img_mask5.shape]

        f3 = rearrange(f3, "b h w d -> b (h w) d")
        img_mask4 = rearrange(img_mask4, "b h w -> b (h w)")

        f4 = rearrange(f4, "b h w d -> b (h w) d")
        img_mask5 = rearrange(img_mask5, "b h w -> b (h w)")

        feature = torch.cat((f3, f4), dim=1)
        mask = torch.cat((img_mask4, img_mask5), dim=1)
        return feature, mask, shapes
    
