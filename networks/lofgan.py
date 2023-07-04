import random

import numpy as np
import torch
from torch import autograd
from torch import nn
from networks.blocks import *
from networks.loss import *
from utils import batched_index_select, batched_scatter
import torch.nn.functional as F
from diffaug import DiffAugment
import torch.nn.utils.spectral_norm as spectral_norm
from thop import profile

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

#AadIN
def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def get_wav(in_channels, pool=True):
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]
    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H
    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)
    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False
    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    return LL, LH, HL, HH


sobel_x = torch.tensor([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
sobel_y = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
laplace = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

def get_wav_two(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
   #  print(filter_LL.size())
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH

def random_mask(img, mask_ratio, mask_size):
    N = img.shape[0]
    device = img.device
    L = mask_size * mask_size # currently we randomly choose patches from the highest resolution
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    mask = torch.ones([N, L], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore).reshape(N, 1, mask_size, mask_size)
    return mask, ids_keep, ids_restore

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
    # # import pdb; pdb.set_trace()
    # loss = (loss * mask).sum() / mask.sum()
    loss = loss.sum() / mask.sum()
    return loss

class WavePool2(nn.Module):
    def __init__(self, in_channels):
        super(WavePool2, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav_two(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav_two(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError

class LoFGAN(nn.Module):
    def __init__(self, config):
        super(LoFGAN, self).__init__()

        self.gen = Generator(config['gen'])
        self.dis = Discriminator(config['dis'])
        self.w_adv_fre = config['w_adv_fre']
        self.w_adv_g = config['w_adv_g']
        self.w_adv_d = config['w_adv_d']
        self.w_recon = config['w_recon']
        self.w_cls = config['w_cls']
        self.w_gp = config['w_gp']
        self.n_sample = config['n_sample_train']
        self.rec_d = config['rec_d']
        self.rec_g = config['rec_g']
        self.w_lap_g = config['w_lap_g']
        self.w_lap_d = config['w_lap_d']
        self.lofgan = config['lofgan']


    def forward(self, xs, y, mode):
        input = torch.rand(8, 1, 3, 128, 128).cuda()
        flops, params = profile(self.gen, inputs=(input,))
        print('FLOPs_G = ' + str(flops / 1000 ** 3) + 'G')
        print('Params_G = ' + str(params / 1000 ** 2) + 'M')
        flops, params = profile(self.dis, inputs=(input,))
        print('FLOPs_D = ' + str(flops / 1000 ** 3) + 'G')
        print('Params_D = ' + str(params / 1000 ** 2) + 'M')
        if mode == 'gen_update':
            if self.lofgan:
                fake_x, similarity, indices_feat, indices_ref, base_index = self.gen(xs)

                loss_recon = local_recon_criterion(xs, fake_x, similarity, indices_feat, indices_ref, base_index, s=8)

                feat_real, _, _, _, _, _ = self.dis(xs)
                feat_fake, logit_adv_fake, logit_c_fake, logit_fre_fake, logit_rec_g, logit_adv_lap = self.dis(fake_x)
                loss_adv_gen = torch.mean(-logit_adv_fake)
                loss_cls_gen = F.cross_entropy(logit_c_fake, y.squeeze())

                if logit_adv_lap is not None:
                    loss_adv_gen_lap = torch.mean(-logit_adv_lap.float())
                else:
                    loss_adv_gen_lap = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)

                if logit_fre_fake is not None:
                    loss_cls_fre = F.cross_entropy(logit_fre_fake, y.squeeze())
                else:
                    loss_cls_fre = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)

                if logit_rec_g is not None:
                    loss_rec_gen = logit_rec_g * self.rec_g
                else:
                    loss_rec_gen = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)

                loss_recon = loss_recon * self.w_recon
                loss_adv_gen = loss_adv_gen * self.w_adv_g
                loss_adv_gen_lap = loss_adv_gen_lap * self.w_lap_g
                loss_cls_gen = loss_cls_gen * self.w_cls
                loss_adv_gen_f = loss_cls_fre * self.w_adv_fre

                loss_total = loss_recon + loss_adv_gen + loss_cls_gen + loss_adv_gen_f + loss_adv_gen_lap
                loss_total.backward()

                return {'loss_total': loss_total,
                    'loss_recon': loss_recon,
                    'loss_adv_gen': loss_adv_gen,
                    'loss_cls_gen': loss_cls_gen}
            else:
                fake_x, similarity, indices_feat, indices_ref, base_index = self.gen(xs)

                loss_recon = local_recon_criterion(xs, fake_x, similarity, indices_feat, indices_ref, base_index, s=8)

                feat_real, _, _, _, _, _ = self.dis(xs, mask=None)
                feat_fake, logit_adv_fake, logit_c_fake, logit_fre_fake, logit_rec_g, logit_adv_lap = self.dis(fake_x, mask=None)
                loss_adv_gen = torch.mean(-logit_adv_fake)

                loss_cls_gen = F.cross_entropy(logit_c_fake, y.squeeze())

                if logit_adv_lap is not None:
                    loss_adv_gen_lap = torch.mean(-logit_adv_lap.float())
                else:
                    loss_adv_gen_lap = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)

                if logit_fre_fake is not None:
                    loss_cls_fre = F.cross_entropy(logit_fre_fake, y.squeeze())
                else:
                    loss_cls_fre = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)

                if logit_rec_g is not None:
                    loss_rec_gen = logit_rec_g * self.rec_g
                else:
                    loss_rec_gen = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)

                loss_recon = loss_recon * self.w_recon
                loss_adv_gen = loss_adv_gen * self.w_adv_g
                loss_adv_gen_lap = loss_adv_gen_lap * self.w_lap_g
                loss_cls_gen = loss_cls_gen * self.w_cls

                loss_adv_gen_f = loss_cls_fre * self.w_adv_fre
                loss_total = loss_recon + loss_adv_gen + loss_cls_gen + loss_rec_gen + loss_adv_gen_f + loss_adv_gen_lap
                # import pdb; pdb.set_trace()
                loss_total.backward()

                return {'loss_total': loss_total,
                    'loss_recon': loss_recon,
                    'loss_adv_gen': loss_adv_gen,
                    'loss_cls_gen': loss_cls_gen}

        elif mode == 'dis_update':
            if self.lofgan:
                xs.requires_grad_()

                _, logit_adv_real, logit_c_real, logit_fre_real, logit_rec, logit_adv_lap = self.dis(xs)
                loss_adv_dis_real = torch.nn.ReLU()(1.0 - logit_adv_real).mean()
                if logit_adv_lap is not None:
                    loss_adv_dis_real_lap = torch.nn.ReLU()(1.0 - logit_adv_lap.float()).mean()
                else:
                    loss_adv_dis_real_lap = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)
                if logit_rec is not None:
                    logit_rec_real = logit_rec * self.rec_d
                else:
                    logit_rec_real = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)
                loss_adv_dis_real = loss_adv_dis_real * self.w_adv_d + logit_rec_real + loss_adv_dis_real_lap * self.w_lap_d
                loss_adv_dis_real.backward(retain_graph=True)

                y_extend = y.repeat(1, self.n_sample).view(-1)
                index = torch.LongTensor(range(y_extend.size(0))).cuda()
                #logit_c_real_forgp = logit_c_real[index, y_extend].unsqueeze(1)
                #loss_reg_dis = self.calc_grad2(logit_c_real_forgp, xs)

                if logit_fre_real is not None:
                    loss_cls_fre = F.cross_entropy(logit_fre_real, y_extend)
                else:
                    loss_cls_fre = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)

               # loss_reg_dis = loss_reg_dis * self.w_gp
               # loss_reg_dis.backward(retain_graph=True)

                loss_cls_dis = F.cross_entropy(logit_c_real, y_extend)
                loss_cls_dis = loss_cls_dis * self.w_cls + loss_cls_fre * self.w_adv_fre
                loss_cls_dis.backward()

                with torch.no_grad():
                    fake_x = self.gen(xs)[0]

                _, logit_adv_fake, _, logit_fre_fake, logit_rec_fake, logit_adv_lap_fake = self.dis(fake_x.detach())
                loss_adv_dis_fake = torch.nn.ReLU()(1.0 + logit_adv_fake).mean()
                if logit_adv_lap_fake is not None:
                    loss_adv_dis_fake_lap = torch.nn.ReLU()(1.0 + logit_adv_lap_fake.float()).mean()
                else:
                    loss_adv_dis_fake_lap = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)
                if logit_rec_fake is not None:
                    loss_rec_fake = self.rec_g * logit_rec_fake
                else:
                    loss_rec_fake = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)
                loss_adv_dis_fake = loss_adv_dis_fake * self.w_adv_d + loss_rec_fake + loss_adv_dis_fake_lap * self.w_lap_g
                loss_adv_dis_fake.backward()

                loss_total = loss_adv_dis_real + loss_adv_dis_fake + loss_cls_dis
                return {'loss_total': loss_total,
                    'loss_adv_dis': loss_adv_dis_fake + loss_adv_dis_real,
                    'loss_adv_dis_real': loss_adv_dis_real,
                    'loss_adv_dis_fake': loss_adv_dis_fake,
                    'loss_cls_dis': loss_cls_dis
                    }
                    #'loss_reg': loss_reg_dis}
            else:
                xs.requires_grad_()
                _, logit_adv_real, logit_c_real, logit_fre_real, logit_rec, logit_adv_lap = self.dis(xs, mask=None)
                loss_adv_dis_real = torch.nn.ReLU()(1.0 - logit_adv_real).mean()
                if logit_adv_lap is not None:
                    loss_adv_dis_real_lap = torch.nn.ReLU()(1.0 - logit_adv_lap.float()).mean()
                else:
                    loss_adv_dis_real_lap = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)
                # # import pdb; pdb.set_trace()
                if logit_rec is not None:
                    logit_rec_real = logit_rec * self.rec_d
                else:
                    logit_rec_real = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)
                loss_adv_dis_real = loss_adv_dis_real * self.w_adv_d + logit_rec_real + loss_adv_dis_real_lap * self.w_lap_d
                loss_adv_dis_real.backward(retain_graph=True)

                y_extend = y.repeat(1, self.n_sample).view(-1).long()
                # index = torch.LongTensor(range(y_extend.size(0))).cuda()
                # logit_c_real_forgp = logit_c_real[index, y_extend].unsqueeze(1)
                # loss_reg_dis = self.calc_grad2(logit_c_real_forgp, xs)

                # loss_reg_dis = loss_reg_dis * self.w_gp
                # loss_reg_dis.backward(retain_graph=True)
                if logit_fre_real is not None:
                    loss_cls_fre = F.cross_entropy(logit_fre_real, y_extend)
                else:
                    loss_cls_fre = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)

                loss_cls_dis = F.cross_entropy(logit_c_real, y_extend)
                loss_cls_dis = loss_cls_dis * self.w_cls + loss_cls_fre * self.w_adv_fre
                # import pdb; pdb.set_trace()
                loss_cls_dis.backward()

                with torch.no_grad():
                    fake_x = self.gen(xs)[0]

                _, logit_adv_fake, _, logit_fre_fake, logit_rec_fake, logit_adv_lap_fake = self.dis(fake_x.detach(), mask=None)
                loss_adv_dis_fake = torch.nn.ReLU()(1.0 + logit_adv_fake).mean()
                if logit_adv_lap_fake is not None:
                    loss_adv_dis_fake_lap = torch.nn.ReLU()(1.0 + logit_adv_lap_fake.float()).mean()
                else:
                    loss_adv_dis_fake_lap = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)
                if logit_rec_fake is not None:
                    loss_rec_fake = self.rec_g * logit_rec_fake
                else:
                    loss_rec_fake = torch.tensor(1.).cuda().requires_grad_(requires_grad=False)
                loss_adv_dis_fake = loss_adv_dis_fake * self.w_adv_d + loss_rec_fake + loss_adv_dis_fake_lap * self.w_lap_g
                loss_adv_dis_fake.backward()

                loss_total = loss_adv_dis_real + loss_adv_dis_fake + loss_cls_dis
                return {'loss_total': loss_total,
                    'loss_adv_dis': loss_adv_dis_fake + loss_adv_dis_real,
                    'loss_adv_dis_real': loss_adv_dis_real,
                    'loss_adv_dis_fake': loss_adv_dis_fake,
                    'loss_cls_dis': loss_cls_dis,
                    'loss_reg': loss_cls_dis}

        else:
            assert 0, 'Not support operation'

    def generate(self, xs):
        fake_x = self.gen(xs)[0]
        return fake_x

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()
        reg /= batch_size
        return reg


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.soft_label = False
        mask_ratio = config['mask_ratio']
        nf = config['nf']
        n_class = config['num_classes']
        n_res_blks = config['n_res_blks']
        mask_rec = config['mask_rec']
        mask_size = config['mask_size']
        patch_size = config['patch_size']
        decoder_embed_dim = config['decoder_embed_dim']
        in_channels = config['in_channels']
        laplace = config['laplace']
        self.fre_loss = config['fre_loss']
        cnn_f = [Conv2dBlock(3, nf, 5, 1, 2,
                             pad_type='reflect',
                             norm='sn',
                             activation='none')]
        for i in range(n_res_blks):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])
        self.laplace = laplace
        if self.laplace:
            cnn_f_lap = [Conv2dBlock(1, nf, 5, 1, 2,
                             pad_type='reflect',
                             norm='sn',
                             activation='none')]
        # for i in range(n_res_blks):
        #     nf_out = np.min([nf * 2, 1024])
        #     cnn_f_lap += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
        #     cnn_f_lap += [nn.ReflectionPad2d(1)]
        #     cnn_f_lap += [nn.AvgPool2d(kernel_size=3, stride=2)]
        #     nf = np.min([nf * 2, 1024])

        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
        # cnn_f_lap += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
        cnn_adv = [nn.AdaptiveAvgPool2d(1),
                   Conv2dBlock(nf_out, n_class, 1, 1,
                               norm='none',
                               activation='none',
                               activation_first=False)]
        if self.laplace:
            cnn_adv_lap = [nn.AdaptiveAvgPool2d(1),
                        Conv2dBlock(nf_out, n_class, 1, 1,
                        norm='none',
                        activation='none',
                        activation_first=False)]
        cnn_c = [nn.AdaptiveAvgPool2d(1),
                 Conv2dBlock(nf_out, n_class, 1, 1,
                             norm='none',
                             activation='none',
                             activation_first=False)]
        if self.fre_loss:
            cnn_fre = [nn.AdaptiveAvgPool2d(1),
                        Conv2dBlock(nf_out, n_class, 1, 1,
                        norm='none',
                        activation='none',
                        activation_first=False)]
        self.pool = WavePool2(1024).cuda()
        self.cnn_f = nn.Sequential(*cnn_f)
        if self.laplace:
            self.cnn_lap = nn.Sequential(*cnn_f_lap)
            self.cnn_adv_lap = nn.Sequential(*cnn_adv_lap)
        self.cnn_adv = nn.Sequential(*cnn_adv)
        self.cnn_c = nn.Sequential(*cnn_c)
        if self.fre_loss:
            self.cnn_fre = nn.Sequential(*cnn_fre)
        self.mask_ratio = mask_ratio
        self.mask_size = mask_size
        self.mask_rec = mask_rec
       # self.laplace = laplace
        self.diffaug = config['diffaug']
        self.policy = config['policy']
       # self.fre_loss = config['fre_loss']
        if self.mask_rec:
            self.patch_size = patch_size
            self.decoder_embed_dim = decoder_embed_dim
            self.proj = nn.Conv2d(in_channels, self.decoder_embed_dim, kernel_size=4, stride=4)
            self.mask_token = torch.nn.Parameter(torch.zeros(1, self.decoder_embed_dim, 1, 1))
            torch.nn.init.normal_(self.mask_token, std=.02)
            self.decoder = nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=1)
            self.pred = nn.Conv2d(self.decoder_embed_dim, self.patch_size*self.patch_size*3, kernel_size=1)

    def conv_operator(self, image, kernel=laplace, in_channels=3, out_channels=1):
        output_laplace = F.conv2d(image, kernel.repeat(out_channels, in_channels, 1, 1).cuda(), stride=1, padding=1, )
        return output_laplace

    def forward(self, x, mask=None):
        if len(x.size()) == 5:
            B, K, C, H, W = x.size()
            x = x.view(B * K, C, H, W)
        else:
            B, C, H, W = x.size()
            K = 1
        if self.diffaug:
            x = DiffAugment(x, policy=self.policy)
        logit_adv_lap = None
        if self.mask_ratio > 0:
            mask, ids_keep, ids_restore  = random_mask(x, mask_ratio=self.mask_ratio, mask_size=self.mask_size)
        if mask is not None:
            # p = int(mask.shape[1] ** 0.5)
            # scale = x.shape[2] // p
            # new_mask = mask.reshape(-1, p, p).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2).unsqueeze(1).type_as(x)
            # x = x * (1. - new_mask)
            new_mask = torch.nn.functional.interpolate(mask, size=(x.shape[2], x.shape[3]), mode='nearest').to(x.dtype)
            x = x * new_mask
        if self.laplace:
            laplace_x = self.conv_operator(x)
            feat_lap = self.cnn_lap(laplace_x)
            logit_adv_lap = self.cnn_adv_lap(feat_lap).view(B * K, -1) + 1e-5
        feat = self.cnn_f(x)
        # frequency analysis
        logit_fre = None
        if self.fre_loss:
            LL, LH, HL, HH = self.pool(feat)
            HF = LH + HL + HH
            logit_fre = self.cnn_fre(HF).view(B * K, -1)
        logit_adv = self.cnn_adv(feat).view(B * K, -1)
        logit_c = self.cnn_c(feat).view(B * K, -1)
        rec_loss = None
        if self.mask_rec and mask is not None:
            feat = self.proj(x)
            n, c, h, w = feat.shape
            temp_mask = torch.stack([new_mask, new_mask, new_mask], dim = 1)
            temp_mask = torch.squeeze(temp_mask, dim=2)
            feat_mask = self.proj(temp_mask)
            mask_token = self.mask_token.repeat(feat.shape[0], 1, feat.shape[2], feat.shape[3])
            feat = feat * (1. - feat_mask) + mask_token * feat_mask
            feat = self.decoder(feat)
            rec = self.pred(feat)
            rec_loss = cal_rec_loss(x, rec, mask, self.patch_size)
        # import pdb; pdb.set_trace()
        return feat, logit_adv, logit_c, logit_fre, rec_loss, logit_adv_lap


class SPADE(nn.Module):
    def __init__(self, feature_size, style_size):
        super(SPADE, self).__init__()
        self.norm = nn.BatchNorm2d(feature_size, affine=False)
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(style_size, 128, 3, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.conv_gamma = spectral_norm(nn.Conv2d(128, feature_size, 3, 1, 1))
        self.conv_beta = spectral_norm(nn.Conv2d(128, feature_size, 3, 1, 1))

    def forward(self, x, s):
        s = F.interpolate(s, size=(x.size(2), x.size(3)), mode='nearest')
        s = self.conv(s)
        return self.norm(x) * self.conv_gamma(s) + self.conv_beta(s)


class SPADEResBlk(nn.Module):
    def __init__(self, input_size, output_size, style_size):
        super(SPADEResBlk, self).__init__()
        # Main layer
        self.spade_1 = SPADE(input_size, style_size)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_1 = spectral_norm(nn.Conv2d(input_size, output_size, 3, 1, 1))
        self.spade_2 = SPADE(output_size, style_size)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv_2 = spectral_norm(nn.Conv2d(output_size, output_size, 3, 1, 1))
        # Shortcut layer
        self.spade_s = SPADE(input_size, style_size)
        self.relu_s = nn.ReLU(inplace=True)
        self.conv_s = spectral_norm(nn.Conv2d(input_size, output_size, 3, 1, 1))

    def forward(self, x, s):
        y = self.conv_1(self.relu_1(self.spade_1(x, s)))
        y = self.conv_2(self.relu_2(self.spade_2(y, s)))
        y_ = self.conv_s(self.relu_s(self.spade_s(x, s)))
        return y + y_

class ModulatedSPADE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero', activation='elu', norm='none', sn=False):
        super(ModulatedSPADE, self).__init__()
        self.out_channels = out_channels

        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
            # self.mask_conv2d = spectral_norm(
                # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            # self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
        self.sigmoid = torch.nn.Sigmoid()

        nhidden = 64
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(in_channels, nhidden, kernel_size=3, stride=stride, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)

        ####### mod 2 ########
        self.mlp_shared_2 = nn.Sequential(
            nn.Conv2d(in_channels, nhidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma_ctx_gamma = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)
        self.mlp_beta_ctx_gamma = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)

        self.mlp_gamma_ctx_beta = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)
        self.mlp_beta_ctx_beta = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2, x3):
        # import pdb; pdb.set_trace()
        x_pad = self.pad(x1)
        conv = self.conv2d(x_pad)
        if self.out_channels == 3:
            return conv

        if self.norm:
            normalized = self.norm(conv)

        ####### mod 2 ########
        ctx = self.mlp_shared_2(x2 + x3)
        gamma_ctx_gamma = self.mlp_gamma_ctx_gamma(ctx)
        beta_ctx_gamma = self.mlp_beta_ctx_gamma(ctx)
        gamma_ctx_beta = self.mlp_gamma_ctx_beta(ctx)
        beta_ctx_beta = self.mlp_beta_ctx_beta(ctx)

        ####### mod 1 ########
        # x_conv = self.conv_x(x)
        actv = self.mlp_shared(x1)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # print(gamma_ctx_gamma.size())
        # print(beta_ctx_gamma.size())
        # print(gamma.size())

        gamma = gamma * (1. + gamma_ctx_gamma) + beta_ctx_gamma
        beta = beta * (1. + gamma_ctx_beta) + beta_ctx_beta
        out_norm = normalized * (1. + gamma) + beta

        if self.activation:
            out = self.activation(out_norm)

        return out

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # # import pdb; pdb.set_trace()
        self.adain = config['adain']
        self.fusion = LocalFusionModule(inplanes=128, rate=config['rate'])
        self.wavegan = config['wavegan']
        self.wavegan_mean = config['wavegan_mean']
        self.spade = config['spade']
        self.spade_block = config['spade_block']
        self.modulated_spade = config['modulated_spade']
        self.K_shot = config['K_shot']
        if self.spade:
            self.spade_layer = SPADE(128, 128)
        if self.spade_block:
            self.spade_blocks = SPADEResBlk(128, 128, 128)
        if self.modulated_spade:
            self.modulated_spade_layer = ModulatedSPADE(128, 128, kernel_size=3, stride=1, padding=1, norm='in', activation='lrelu')

    def forward(self, xs):
        b, k, C, H, W = xs.size()
        xs = xs.view(-1, C, H, W)
        # print("encoder")
        if self.wavegan:
            pool_flag = 1
        else:
            pool_flag = 0
        querys, skips = self.encoder(xs, pool_flag)
        c, h, w = querys.size()[-3:]
        querys = querys.view(b, k, c, h, w)

        similarity_total = torch.cat([torch.rand(b, 1) for _ in range(k)], dim=1).cuda()  # b*k
        similarity_sum = torch.sum(similarity_total, dim=1, keepdim=True).expand(b, k)  # b*k
        similarity = similarity_total / similarity_sum  # b*k

        
        if k == 1:
            base_feat = querys
            base_index = 0
            feat_gen = querys[:, base_index, :, :, :]
            indices_feat = None
            indices_ref = None
        else:
            base_index = random.choice(range(k))
            # # import pdb; pdb.set_trace()
            base_feat = querys[:, base_index, :, :, :]
            feat_gen, indices_feat, indices_ref = self.fusion(base_feat, querys, base_index, similarity)
        if self.adain:
            base_index_adain = k - base_index - 1
            style_feat = querys[:, base_index_adain, :, :, :]
            feat_gen = torch.tensor(adaptive_instance_normalization(feat_gen, style_feat))
        if self.spade:
            base_index_spade = k - base_index - 1
            style_feat = querys[:, base_index_spade, :, :, :]
            feat_gen = self.spade_layer(feat_gen, style_feat)
        if self.spade_block:
            base_index_spade = k - base_index - 1
            style_feat = querys[:, base_index_spade, :, :, :]
            feat_gen = self.spade_blocks(feat_gen, style_feat)
        if self.modulated_spade:
            base_index_spade = k - base_index - 1
            style_feat_1 = querys[:, base_index_spade, :, :, :]
            style_feat_2 = querys[:, base_index_spade - 1, :, :, :]
            feat_gen = self.modulated_spade_layer(feat_gen, style_feat_1, style_feat_2)





        fake_x = self.decoder(feat_gen, skips, self.wavegan, self.wavegan_mean, base_index, self.K_shot)

        return fake_x, similarity, indices_feat, indices_ref, base_index


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = Conv2dBlock(3, 32, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.pool1 = WavePool(32).cuda()
        self.conv2 = Conv2dBlock(32, 64, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.pool2 = WavePool(64).cuda()
        self.conv3 = Conv2dBlock(64, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.pool3 = WavePool2(128).cuda()
        self.conv4 = Conv2dBlock(128, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.pool4 = WavePool2(128).cuda()
        self.conv5 = Conv2dBlock(128, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')

    def forward(self, x, pool_flag):
        #(24,3,128,128)
        skips = {}
        x = self.conv1(x)  #(24,32,128,128)
        skips['conv1_1'] = x
        LL1, LH1, HL1, HH1 = self.pool1(x) # (24,64,64,64)
        skips['pool1'] = [LH1, HL1, HH1]
        x = self.conv2(x)    #(24,64,64,64)
        # p2 = self.pool2(x)  #24,128,32,32
        skips['conv2_1'] = x
        LL2, LH2, HL2, HH2 = self.pool2(x)  #24,128,32,32
        skips['pool2'] = [LH2, HL2, HH2]
        if pool_flag:
            x = self.conv3(x+LL1)
        else:
            x = self.conv3(x)         #(24,128,32,32)
        # p3 = self.pool3(x)
        skips['conv3_1'] = x
        LL3, LH3, HL3, HH3 = self.pool3(x)
        #(24,128,16,16)
        skips['pool3'] = [LH3, HL3, HH3]
        #(24,128,32,32)
        if pool_flag:
            x = self.conv4(x+LL2)        #(24,128,16,16)
        else:
            x = self.conv4(x)
        skips['conv4_1'] = x
        LL4, LH4, HL4, HH4 = self.pool4(x)
        skips['pool4'] = [LH4, HL4, HH4]        #(24,128,8,8)
        if pool_flag:
            x = self.conv5(x+LL3)        #(24,128,8,8)
        else:
            x = self.conv5(x)
        return x, skips


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.Upsample = nn.Upsample(scale_factor=2)
        self.Conv1 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block1 = WaveUnpool(128,"sum").cuda()
        self.Conv2 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block2 = WaveUnpool(128, "sum").cuda()
        self.Conv3 = Conv2dBlock(128, 64, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block3 = WaveUnpool(64, "sum").cuda()
        self.Conv4 = Conv2dBlock(64, 32, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block4 = WaveUnpool(32, "sum").cuda()
        self.Conv5 = Conv2dBlock(32, 3, 5, 1, 2,
                             norm='none',
                             activation='tanh',
                             pad_type='reflect')

    def forward(self, x, skips, wavegan, wavegan_mean, base_index, K_shot):
        x1 = self.Upsample(x)
        x2 = self.Conv1(x1)
        if wavegan:
            LH1, HL1, HH1 = skips['pool4']
            c, h, w = LH1.size()[-3:]
            if K_shot == 1:
                LH1, HL1, HH1 = LH1.view(8, 1,c, h, w).squeeze(), HL1.view(8, 1,c, h, w).squeeze(), HH1.view(8, 1,c, h, w).squeeze()
            else:
                if wavegan_mean:
                    LH1, HL1, HH1 = LH1.view(8, K_shot, c, h, w).mean(dim=1), HL1.view(8, K_shot, c, h, w).mean(dim=1), HH1.view(8, K_shot, c, h, w).mean(dim=1)
                else:
                    LH1, HL1, HH1 = LH1.view(8, K_shot, c, h, w), HL1.view(8, K_shot, c, h, w), HH1.view(8, K_shot, c, h, w)
                    LH1, HL1, HH1 = LH1[:,base_index,:,:,:], HL1[:,base_index,:,:,:], HH1[:,base_index,:,:,:]
            original1 = skips['conv4_1']
            x_deconv = self.recon_block1(x, LH1, HL1, HH1, original1)
            x2 = x_deconv + x2
            x3 = self.Upsample(x2)
            x4 = self.Conv2(x3)
            LH2, HL2, HH2 = skips['pool3']
            original2 = skips['conv3_1']
            c, h, w = LH2.size()[-3:]
            if K_shot == 1:
                LH2, HL2, HH2 = LH2.view(8, 1, c, h, w).squeeze(), HL2.view(8, 1, c, h, w).squeeze(), HH2.view(8, 1, c, h, w).squeeze()
            else:
                if wavegan_mean:
                    LH2, HL2, HH2 = LH2.view(8, K_shot, c, h, w).mean(dim=1), HL2.view(8, K_shot, c, h, w).mean(dim=1), HH2.view(8, K_shot, c, h, w).mean(dim=1)
                else:
                    LH2, HL2, HH2 = LH2.view(8, K_shot, c, h, w), HL2.view(8, K_shot, c, h, w), HH2.view(8, K_shot, c, h, w)
                    LH2, HL2, HH2 = LH2[:, base_index, :, :, :], HL2[:, base_index, :, :, :], HH2[:, base_index, :, :, :]
            x_deconv2 = self.recon_block1(x1, LH2, HL2, HH2, original2)
            LH3, HL3, HH3 = skips['pool2']
            c, h, w = skips['conv2_1'].size()[-3:]
            original3 = skips['conv2_1']
            c, h, w = LH3.size()[-3:]

            if K_shot == 1:
                LH3, HL3, HH3 = LH3.view(8, 1, c, h, w).squeeze(), HL3.view(8, 1, c, h, w).squeeze(), HH3.view(8, 1, c, h, w).squeeze()
            else:
                if wavegan_mean:
                    LH3, HL3, HH3 = LH3.view(8, K_shot, c, h, w).mean(dim=1), HL3.view(8, K_shot, c, h, w).mean(dim=1), HH3.view(8, K_shot, c, h, w).mean(dim=1)
                else:
                    LH3, HL3, HH3 = LH3.view(8, K_shot, c, h, w), HL3.view(8, K_shot, c, h, w), HH3.view(8, K_shot, c, h, w)
                    LH3, HL3, HH3 = LH3[:, base_index, :, :, :], HL3[:, base_index, :, :, :], HH3[:, base_index, :, :, :]
            x_deconv4 = self.recon_block1(x3, LH3, HL3, HH3, original2)
            x5 = self.Upsample(x4+x_deconv2)
            x6 = self.Conv3(x5+x_deconv4)

            LH4, HL4, HH4 = skips['pool1']
            original4 = skips['conv1_1']
            c, h, w = LH4.size()[-3:]
            if K_shot == 1:
                LH4, HL4, HH4 = LH4.view(8, 1, c, h, w).squeeze(), HL4.view(8, 1, c, h, w).squeeze(), HH4.view(8, 1, c, h, w).squeeze()
            else:
                if wavegan_mean:
                    LH4, HL4, HH4 = LH4.view(8, K_shot, c, h, w).mean(dim=1), HL4.view(8, K_shot, c, h, w).mean(dim=1), HH4.view(8, K_shot, c, h, w).mean(dim=1)
                else:
                    LH4, HL4, HH4 = LH4.view(8, K_shot, c, h, w), HL4.view(8, K_shot, c, h, w), HH4.view(8, K_shot, c, h, w)
                    LH4, HL4, HH4 = LH4[:, base_index, :, :, :], HL4[:, base_index, :, :, :], HH4[:, base_index, :, :, :]
            x_deconv3 = self.recon_block3(x6, LH4, HL4, HH4, original3)
            x7 = self.Upsample(x6)
            x8 = self.Conv4(x7 + x_deconv3)
            x9 = self.Conv5(x8)
        else:
            x3 = self.Upsample(x2)
            x4 = self.Conv2(x3)
            x5 = self.Upsample(x4)
            x6 = self.Conv3(x5)
            x7 = self.Upsample(x6)
            x8 = self.Conv4(x7)
            x9 = self.Conv5(x8)

        return x9



class LocalFusionModule(nn.Module):
    def __init__(self, inplanes, rate):
        super(LocalFusionModule, self).__init__()

        self.W = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inplanes)
        )
        self.rate = rate

    def forward(self, feat, refs, index, similarity):


        # if n == 1:
        #     feat_indices = torch.cat([torch.LongTensor(random.sample(range(h * w), num)).unsqueeze(0) for _ in range(b)],dim=0).cuda()  # B*num
        #     import pdb; pdb.set_trace()
        #     return feat[:,index,:,:,:], 
        refs = torch.cat([refs[:, :index, :, :, :], refs[:, (index + 1):, :, :, :]], dim=1)
        base_similarity = similarity[:, index]
        ref_similarities = torch.cat([similarity[:, :index], similarity[:, (index + 1):]], dim=1)
        b, n, c, h, w = refs.size()
        # take ref:(32, 2, 128, 8, 8) for example
        refs = refs.view(b * n, c, h, w)

        w_feat = feat.view(b, c, -1)
        w_feat = w_feat.permute(0, 2, 1).contiguous()
        w_feat = F.normalize(w_feat, dim=2)  # (32*64*128)

        w_refs = refs.view(b, n, c, -1)
        w_refs = w_refs.permute(0, 2, 1, 3).contiguous().view(b, c, -1)
        w_refs = F.normalize(w_refs, dim=1)  # (32*128*128)

        # local selection
        rate = self.rate
        num = int(rate * h * w)
        feat_indices = torch.cat([torch.LongTensor(random.sample(range(h * w), num)).unsqueeze(0) for _ in range(b)],dim=0).cuda()  # B*num

        feat = feat.view(b, c, -1)  # (32*128*64)
        feat_select = batched_index_select(feat, dim=2, index=feat_indices)  # (32*128*12)

        # local matching
        w_feat_select = batched_index_select(w_feat, dim=1, index=feat_indices)  # (32*12*128)
        w_feat_select = F.normalize(w_feat_select, dim=2)  # (32*12*128)

        refs = refs.view(b, n, c, h * w)
        ref_indices = []
        ref_selects = []
        for j in range(n):
            ref = refs[:, j, :, :]  # (32*128*64)
            w_ref = w_refs.view(b, c, n, h * w)[:, :, j, :]  # (32*128*64)
            fx = torch.matmul(w_feat_select, w_ref)  # (32*12*64)
            _, indice = torch.topk(fx, dim=2, k=1)
            indice = indice.squeeze(0).squeeze(-1)  # (32*10)
            select = batched_index_select(ref, dim=2, index=indice)  # (32*128*12)
            ref_indices.append(indice)
            ref_selects.append(select)
        ref_indices = torch.cat([item.unsqueeze(1) for item in ref_indices], dim=1)  # (32*2*12)
        ref_selects = torch.cat([item.unsqueeze(1) for item in ref_selects], dim=1)  # (32*2*128*12)

        # local replacement
        base_similarity = base_similarity.view(b, 1, 1)  # (32*1*1)
        ref_similarities = ref_similarities.view(b, 1, n)  # (32*1*2)
        feat_select = feat_select.view(b, 1, -1)  # (32*1*(128*12))
        ref_selects = ref_selects.view(b, n, -1)  # (32*2*(128*12))

        feat_fused = torch.matmul(base_similarity, feat_select) \
                     + torch.matmul(ref_similarities, ref_selects)  # (32*1*(128*12))
        feat_fused = feat_fused.view(b, c, num)  # (32*128*12)

        feat = batched_scatter(feat, dim=2, index=feat_indices, src=feat_fused)
        feat = feat.view(b, c, h, w)  # (32*128*8*8)

        return feat, feat_indices, ref_indices  # (32*128*8*8), (32*12), (32*2*12)


