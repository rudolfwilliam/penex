"""Copied from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py"""

from torch import nn
import torch.nn.utils as U
import torch.nn.functional as F
import torch
import kornia.augmentation as K
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from project_files.vision.forward_modules import ForwardModule

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class ViT(ForwardModule):
    def __init__(
            self,
            image_size,
            patch_size,
            num_classes,
            dim,
            depth,
            heads, 
            mlp_dim, 
            pool = 'cls', 
            channels = 3, 
            dim_head = 64, 
            dropout = 0., 
            emb_dropout = 0.,
            deep_supervision=False
            ):
        
        assert deep_supervision is False, "Deep supervision not supported for ViT."

        super().__init__(n_out=num_classes, deep_supervision=deep_supervision)

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))


        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self._init_weights()

        """if augment:
            self.augment = Augment()
        else:
            self.augment = nn.Identity()"""
    
    def _init_weights(self):
        # Initialize position embeddings
        torch.nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, img):
        #img = self.augment(img)  # Apply augmentations if enabled
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        final_out = self.mlp_head(x)

        return final_out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


"""class Augment(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), mode="imagenet"):
        assert mode in ["imagenet"], "Only imagenet mode is supported currently."
        super().__init__()
        self.mean = mean
        self.std = std

        self.train_aug = K.AugmentationSequential(
            K.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),  # Better for ImageNet
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(0.1, 0.1, 0.1, 0.05, p=0.5),  # Reduced intensity
            # Don't normalize here - already done in transforms
            K.RandomErasing(p=0.1, scale=(0.02, 0.10), ratio=(0.3, 3.3)),  # Reduced probability
            data_keys=["input"],
        )

    def forward(self, x):
        if self.training:
            # x is already normalized from your transforms
            x = self._unnormalize(x)  # Convert back to [0,1]
            x = self.train_aug(x)     # Apply augmentations
            x = self._normalize(x)    # Convert back to normalized
            return x
        else:
            return x
    
    def _unnormalize(self, x):
        mean = x.new_tensor(self.mean).view(1, -1, 1, 1)
        std  = x.new_tensor(self.std).view(1, -1, 1, 1)
        return x * std + mean

    def _normalize(self, x):
        mean = x.new_tensor(self.mean).view(1, -1, 1, 1)
        std  = x.new_tensor(self.std).view(1, -1, 1, 1)
        return (x - mean) / std"""
        