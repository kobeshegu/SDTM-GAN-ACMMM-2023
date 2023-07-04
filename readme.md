## Official codes for "Improving Few-shot Image Generation by Structural Discrimination and Textural Modulation"

### Requirements
```
    imageio==2.9.0
    lmdb==1.2.1
    opencv-python==4.5.3
    pillow==8.3.2
    scikit-image==0.17.2
    scipy==1.5.4
    tensorboard==2.7.0
    tensorboardx==2.4
    torch==1.7.0+cu110
    torchvision==0.8.1+cu110
    tqdm==4.62.3
    pytorch-fid
    clean-fid
    prdc
```

## Training:
First, prepare the datasets from the repo of LofGAN and put them in the datasets folder.

Then set the parameters for our proposed TexMod and StructD in ./configs, to reproduce our results on WaveGAN, you need:
```
    # logger options
snapshot_save_iter: 20000
snapshot_val_iter: 2000
snapshot_log_iter: 200

# optimization options
max_iter: 100000
weight_decay: 0.0001
lr_gen: 0.0001
lr_dis: 0.0001
init: kaiming
w_adv_g: 1
w_adv_d: 1
w_lap_g: 1
w_lap_d: 1
w_recon: 0.5
w_cls: 1.0
w_gp: 10
rec_d: 1
rec_g: 1
w_adv_fre: 1
lofgan: False

# model options
model: LoFGAN
gen:
  nf: 32
  n_downs: 4
  norm: bn
  rate: 0.5
  wavegan: True
  wavegan_mean: False
  adain: False
  spade: False
  spade_block: False
  modulated_spade: True
  K_shot: 3
dis:
  nf: 64
  n_res_blks: 4
  num_classes: 119
  mask_ratio: 0
  mask_size: 4
  mask_rec : False
  patch_size : 4
  decoder_embed_dim : 128
  in_channels : 3
  laplace: True
  diffaug: False
  policy: 'color,translation,cutout'
  fre_loss: True

# data options
dataset: animal
num_workers: 8
batch_size: 8
n_sample_train: 9
n_sample_val: 9
n_sample_test: 9
num_generate: 10
data_root: datasets/animal_128.npy      
```

Finally, run:
`bash scripts/train.sh`
The quantitative and qualitative results will be automatically saved in /results folder.

