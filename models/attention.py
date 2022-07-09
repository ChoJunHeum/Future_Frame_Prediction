import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Attention(nn.Module):
    def __init__(self, dim, image_size, patch_size, heads = 8, dim_head = 64, dropout = 0., channels = 3,):
        super().__init__()
        inner_dim = dim_head *  heads

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        project_out = not (heads == 1 and dim_head == dim)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.embeddings = torch.Tensor([])
        self.QKVS = torch.Tensor([])

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, images):
        embeddings = torch.Tensor([])

        for image in images:
            x = self.to_patch_embedding(image)
            qkv = self.to_qkv(x).chunk(3, dim = -1)
            self.embeddings = torch.cat((embeddings, x), dim=1)

        print(self.embeddings.shape)
        quit()

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def _test():
    rand = torch.ones([1, 12, 256, 256]).cuda()
    t = Attention(
        image_size = 256,
        patch_size = 32,
        num_classes = 10,
        dim = 1024,
        depth = 9,
        heads = 16,
        mlp_dim = 2048
    )
    r = t(rand)
    print(r.shape)
    print(r.grad_fn)
    print(r.requires_grad)

_test()