import torch
from torch import nn

# def random_mask(img, mask_ratio):
#     N = img.shape[0]
#     device = img.device
#     L = 4 * 4 # currently we randomly choose patches from the highest resolution
#     len_keep = int(L * (1 - mask_ratio))
#     noise = torch.rand(N, L, device=device)  # noise in [0, 1]
#     ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#     ids_restore = torch.argsort(ids_shuffle, dim=1)
#     ids_keep = ids_shuffle[:, :len_keep]
#     mask = torch.ones([N, L], device=device)
#     mask[:, :len_keep] = 0
#     mask = torch.gather(mask, dim=1, index=ids_restore)
#     return mask

def random_mask(img, mask_ratio):
    N = img.shape[0]
    device = img.device
    L = 4 * 4 # currently we randomly choose patches from the highest resolution
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    # generate the binary mask: 0 is keep 1 is remove
    mask = torch.ones([N, L], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask

def patchify(imgs, patch_size=32):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w * 3, p**2))
    return x

def cal_rec_loss(imgs, pred, mask, patch_size):
    target = patchify(imgs, patch_size)
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    target = (target - mean) / (var + 1.e-6)**.5
    if len(pred.shape) == 4:
        n, c, _ = target.shape
        pred = pred.reshape(n, c, -1)
        # pred = torch.einsum('ncl->nlc', pred)
    loss = (pred - target) ** 2
    loss = loss.mean(dim=1)  # [N, L], mean loss per patch
    loss = (loss * mask).sum() / mask.sum()
    return loss


patch_size = 4
decoder_embed_dim = 128
in_channels = 3
proj = nn.Conv2d(in_channels, decoder_embed_dim, kernel_size=4, stride=4).cuda()
mask_token = torch.nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1)).cuda()
torch.nn.init.normal_(mask_token, std=.02)
decoder = nn.Conv2d(decoder_embed_dim, decoder_embed_dim, kernel_size=1).cuda()
pred = nn.Conv2d(decoder_embed_dim, patch_size * patch_size * 3, kernel_size=1).cuda()

imgs = torch.randn([24,3,128,128]).cuda()
mask = random_mask(imgs, mask_ratio=0.02)
p = int(mask.shape[1] ** 0.5)
scale = imgs.shape[2] // p
new_mask = mask.reshape(-1, p, p).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2).unsqueeze(1).type_as(imgs)

target = patchify(imgs, patch_size)
mean = target.mean(dim=-1, keepdim=True)
var = target.var(dim=-1, keepdim=True)
target = (target - mean) / (var + 1.e-6) ** .5
imgs = imgs * (1. - new_mask)

feat = proj(imgs)
temp_mask = torch.stack([new_mask, new_mask, new_mask], dim = 1)
temp_mask = torch.squeeze(temp_mask, dim=2)
feat_mask = proj(temp_mask)
# n, c, h, w = feat.shape
# temp_mask = new_mask.reshape(24, -1, h, w).type_as(feat)
mask_token = mask_token.repeat(feat.shape[0], 1, feat.shape[2], feat.shape[3])
feat = feat * (1. - feat_mask) + mask_token * feat_mask
feat = decoder(feat)
rec = pred(feat)

cal_rec_loss(imgs, rec, mask, patch_size=patch_size)