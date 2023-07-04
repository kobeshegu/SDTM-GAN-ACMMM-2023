import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import inception_v3, Inception3
from torchvision.utils import save_image

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import numpy as np
from scipy import linalg
from tqdm import tqdm
import pickle
import os

import os
import random
import shutil

import cv2
import lpips
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import torch.utils.data
import torchvision.transforms as transforms
from trainer_metric import Trainer
from utils import get_config, unloader, get_model_list
from cleanfid import fid as cleanfid
from prdc import compute_prdc

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 use_fid_inception=True):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def fid_inception_v3():
    """Build pretrained Inception model for FID computation
    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.
    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = models.inception_v3(num_classes=1008,
                                    aux_logits=False,
                                    pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Inception3Feature(Inception3):
    def forward(self, x):
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)

        x = self.Conv2d_1a_3x3(x)  # 299 x 299 x 3
        x = self.Conv2d_2a_3x3(x)  # 149 x 149 x 32
        x = self.Conv2d_2b_3x3(x)  # 147 x 147 x 32
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 147 x 147 x 64

        x = self.Conv2d_3b_1x1(x)  # 73 x 73 x 64
        x = self.Conv2d_4a_3x3(x)  # 73 x 73 x 80
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 71 x 71 x 192

        x = self.Mixed_5b(x)  # 35 x 35 x 192
        x = self.Mixed_5c(x)  # 35 x 35 x 256
        x = self.Mixed_5d(x)  # 35 x 35 x 288

        x = self.Mixed_6a(x)  # 35 x 35 x 288
        x = self.Mixed_6b(x)  # 17 x 17 x 768
        x = self.Mixed_6c(x)  # 17 x 17 x 768
        x = self.Mixed_6d(x)  # 17 x 17 x 768
        x = self.Mixed_6e(x)  # 17 x 17 x 768

        x = self.Mixed_7a(x)  # 17 x 17 x 768
        x = self.Mixed_7b(x)  # 8 x 8 x 1280
        x = self.Mixed_7c(x)  # 8 x 8 x 2048

        x = F.avg_pool2d(x, kernel_size=8)  # 8 x 8 x 2048

        return x.view(x.shape[0], x.shape[1])  # 1 x 1 x 2048


def load_patched_inception_v3():
    # inception = inception_v3(pretrained=True)
    # inception_feat = Inception3Feature()
    # inception_feat.load_state_dict(inception.state_dict())
    inception_feat = InceptionV3([3], normalize_input=False)

    return inception_feat


@torch.no_grad()
def extract_features(loader, inception, device):
    pbar = tqdm(loader)

    feature_list = []

    for img in pbar:
        img = img.to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

    features = torch.cat(feature_list, 0)

    return features


@torch.no_grad()
def extract_feature_from_samples(generator, inception, device='cuda'):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 512, device=device)
        img, _ = g([latent], truncation=truncation, truncation_latent=truncation_latent)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)

    return features


@torch.no_grad()
def extract_feature_from_generator_fn(generator_fn, inception, device='cuda', total=1000):
    features = []
    for batch in tqdm(generator_fn, total=total):
        feat = inception(batch)[0].view(batch.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0).detach()
    return features.numpy()


def calc_fid(sample_features, real_features=None, real_mean=None, real_cov=None, eps=1e-6):
    sample_mean = np.mean(sample_features, 0)
    sample_cov = np.cov(sample_features, rowvar=False)

    if real_features is not None:
        real_mean = np.mean(real_features, 0)
        real_cov = np.cov(real_features, rowvar=False)

    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,default="results/flower_frecls_1_diffaug_cutout_translation_norec")
parser.add_argument('--dataset', type=str, default="flower")
parser.add_argument('--real_dir', type=str, default="results/flower_frecls_1_diffaug_cutout_translation_norec/reals")
parser.add_argument('--fake_dir', type=str,default="results/flower_frecls_1_diffaug_cutout_translation_norec/tests")
parser.add_argument('--ckpt', type=str, default="gen_00100000.pt")
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--n_sample_test', type=int, default=3)
args = parser.parse_args()

conf_file = os.path.join(args.name, 'configs.yaml')
config = get_config(conf_file)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)


if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    real_dir = args.real_dir
    fake_dir = args.fake_dir
    print('real dir: ', real_dir)
    print('fake dir: ', fake_dir)

    # if os.path.exists(fake_dir):
    #     shutil.rmtree(fake_dir)
    # os.makedirs(fake_dir, exist_ok=True)

    # data = np.load(config['data_root'])
    # if args.dataset == 'flower':
    #     data = data[85:]
    #     num = 10
    # elif args.dataset == 'animal':
    #     data = data[119:]
    #     num = 10
    # elif args.dataset == 'vggface':
    #     data = data[1802:]
    #     num = 30

    # data_for_gen = data[:, :num, :, :, :]
    # data_for_fid = data[:, num:, :, :, :]

    # if not os.path.exists(real_dir):
    #     os.makedirs(real_dir, exist_ok=True)
    #     for cls in tqdm(range(data_for_fid.shape[0]), desc='preparing real images'):
    #         for i in range(data_for_fid.shape[1]):
    #             idx = i
    #             real_img = data_for_fid[cls, idx, :, :, :]
    #             if args.dataset == 'vggface':
    #                 real_img *= 255
    #             real_img = Image.fromarray(np.uint8(real_img))
    #             real_img.save(os.path.join(real_dir, '{}_{}.png'.format(cls, str(i).zfill(3))), 'png')

    # if os.path.exists(fake_dir):
    #     trainer = Trainer(config)
    #     if args.ckpt:
    #         last_model_name = os.path.join(args.name, 'checkpoints', args.ckpt)
    #     else:
    #         last_model_name = get_model_list(os.path.join(args.name, 'checkpoints'), "gen")
    #     trainer.load_ckpt(last_model_name)
    #     trainer.cuda()
    #     trainer.eval()
    #     for cls in tqdm(range(data_for_gen.shape[0]), desc='generating fake images'):
    #         for i in range(128):
    #             idx = np.random.choice(data_for_gen.shape[1], args.n_sample_test)
    #             imgs = data_for_gen[cls, idx, :, :, :]
    #             imgs = torch.cat([transform(img).unsqueeze(0) for img in imgs], dim=0).unsqueeze(0).cuda()
    #             fake_x = trainer.generate(imgs)
    #             output = unloader(fake_x[0].cpu())
    #             output.save(os.path.join(fake_dir, '{}_{}.png'.format(cls, str(i).zfill(3))), 'png')

    # fid(real_dir, fake_dir, args.gpu)
    # LPIPS(fake_dir)
    output_name = args.name + '/results.txt'
    f = open(output_name, 'w')

    from torch.utils.data import DataLoader
    from torchvision import utils as vutils

    IM_SIZE = 1024
    BATCH_SIZE = 16
    DATALOADER_WORKERS = 8
    NBR_CLS = 2000
    TRIAL_NAME = 'trial_vae_512_1'
    SAVE_FOLDER = './'

    from torchvision.datasets import ImageFolder

    def real_image_loader(dataloader, n_batches=10):
        counter = 0
        while counter < n_batches:
            counter += 1
            try:
                rgb_img, _ = next(dataloader)
            except StopIteration:
                pass
            if counter == 1:
                vutils.save_image(0.5 * (rgb_img + 1), 'tmp_real.jpg')
            yield rgb_img.cuda()

    inception = load_patched_inception_v3().cuda()
    inception.eval()

    path_a = real_dir
    path_b = fake_dir

    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            # transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dset_a = ImageFolder(path_a, transform)
    loader_a = iter(DataLoader(dset_a, batch_size=16, num_workers=4))

    real_features = extract_feature_from_generator_fn(
        real_image_loader(loader_a, n_batches=900), inception)
    real_mean = np.mean(real_features, 0)
    real_cov = np.cov(real_features, rowvar=False)
    dset_b = ImageFolder(path_b, transform)
    loader_b = iter(DataLoader(dset_b, batch_size=16, num_workers=4))

    sample_features = extract_feature_from_generator_fn(
        real_image_loader(loader_b, n_batches=900), inception)
    # sample_features = extract_feature_from_generator_fn(
    #        image_generator(dataset, net_ae, net_ig, n_batches=1800), inception,
    #         total=1800 )

    # fid = calc_fid(sample_features, real_mean=real_features['mean'], real_cov=real_features['cov'])
    nearest_k = 5
    metrics = compute_prdc(real_features=real_features,
                           fake_features=sample_features,
                           nearest_k=nearest_k)
    print(metrics)
    for key, value in metrics.items():
        f.write(key + ':' + str(value))
        f.write('\n')
    clean_fid = cleanfid.compute_fid(path_a, path_b, mode="clean", num_workers=0)
    clean_kid = cleanfid.compute_kid(path_a, path_b, mode="clean", num_workers=0)
    print("clean_fid:%.10f\n" % clean_fid)
    print("clean_kid:%.10f\n" % clean_kid)
    f.writelines('%s:%.10f\n' % ("clean_fid:", clean_fid))
    f.writelines('%s:%.10f\n' % ("clean_kid:", clean_kid))
    fid = calc_fid(sample_features, real_mean=real_mean, real_cov=real_cov)

    fid = calc_fid(sample_features, real_mean=real_mean, real_cov=real_cov)
    print(fid)
    f.writelines('%s:%.10f\n' % ("FID:", fid))
    f.writelines('%s\n' % ("--------------------the is the split line---------------------"))
    f.close()

