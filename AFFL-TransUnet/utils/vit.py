import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2) #16

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)# 16,257,3072
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)# 16,257,1024

    def forward(self, x, mask=None): # 16,257,1024
        qkv = self.qkv_layer(x)# 16,257,3072

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))#16,4,257,256
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk #16,4,257,257

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)#16,4,257,257

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)#16,4,257,256

        x = rearrange(x, "b h t d -> b t (h d)")#16,257,1024
        x = self.out_attention(x)#16,257,1024

        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module): #1024,4,256
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)# 1024,256

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):#16,257,1024
        _x = self.multi_head_attention(x)#16,257,1024
        _x = self.dropout(_x)#16,257,1024
        x = x + _x #16,257,1024
        x = self.layer_norm1(x)#16,257,1024

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module): #1024,4,256,8
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):# 16,257,1024
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x


class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim, # 16, 1024, 1024, 4, 256
                 block_num, patch_dim, classification=True, num_classes=1): # 8, 1, False, 4
        super().__init__()

        self.patch_dim = patch_dim
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2 #256
        self.token_dim = in_channels * (patch_dim ** 2) #1024

        self.projection = nn.Linear(self.token_dim, embedding_dim) #1024,1024
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim)) # 257, 1024

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))# 1,1,1024

        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num) #1024,4,256,8

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        img_patches = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',#16, 256, 1024
                                patch_x=self.patch_dim, patch_y=self.patch_dim)

        batch_size, tokens, _ = img_patches.shape

        project = self.projection(img_patches)#16, 256, 1024
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size)# 16, 1, 1024

        patches = torch.cat([token, project], dim=1)# 16,257,1024
        patches += self.embedding[:tokens + 1, :]# 16,257,1024

        x = self.dropout(patches)# 16,257,1024
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]

        return x


if __name__ == '__main__':
    vit = ViT(img_dim=256,#128 original
              in_channels=3,
              patch_dim=2,
              embedding_dim=512,
              block_num=2,
              head_num=2,
              mlp_dim=1024)
    print(sum(p.numel() for p in vit.parameters()))
    print(vit(torch.rand(1, 3, 256, 256)).shape)
