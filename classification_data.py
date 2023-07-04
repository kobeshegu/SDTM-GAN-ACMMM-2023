import os
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import argparse
import torch
# from torch.nn.functional import F
def moveFile(fileDir, trainDir, dataset='animal', generate_data=None):

    if dataset == 'flower':
        picknumber = 10
    elif dataset == 'animal':
        picknumber = 35
    elif dataset == 'vggface':
        picknumber = 30

    if generate_data is None:
        for imgpath in os.listdir(fileDir):
            print("------")
            imgdir = os.path.join(fileDir, imgpath)
            pathDir = os.listdir(imgdir)
            sample1 = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本
            save_dir = os.path.join(trainDir, imgpath)
            os.makedirs(save_dir, exist_ok=True)
            print(sample1)
            for name in sample1:
                shutil.move(imgdir + '/' + name, save_dir + "\\" + name)
    else:

        for imgpath in os.listdir(fileDir):
            imgdir = os.path.join(fileDir, imgpath)
            pathDir = os.listdir(imgdir)
            sample1 = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本
            save_dir = os.path.join(trainDir, 'debug', imgpath)
            os.makedirs(save_dir, exist_ok=True)
            print(sample1)
            for name in sample1:
                shutil.copyfile(imgdir + '/' + name, save_dir + "\\" + name)
        # copy train images from original test_images
        saved_dir = os.path.join(trainDir, 'debug')
        for imgpath in os.listdir(saved_dir):
            imgdir = os.path.join(saved_dir, imgpath)
            pathDir = os.listdir(imgdir)
            filenumber = len(pathDir)
            sample1 = random.sample(pathDir, filenumber)  # 随机选取picknumber数量的样本
            save_dir_gen = os.path.join(trainDir, 'generate', imgpath)
            os.makedirs(save_dir_gen, exist_ok=True)
            print(sample1)
            for name in sample1:

                shutil.copyfile(imgdir + '/' + name, save_dir_gen + "\\" + name)
        save_dir_gen = os.path.join(trainDir, 'test2')
        os.makedirs(save_dir_gen, exist_ok=True)
        for imgpath in os.listdir(saved_dir):
            for img in os.listdir(generate_data):
                picknumber = random.sample(range(128), 30)
                for name in picknumber:
                    str1 = '_0' + str(name) + '.png'
                    str2 = '_' + str(name) + '.png'
                    if dataset == 'flower':
                        class_index = 85
                    elif dataset == 'animal':
                        class_index = 119
                    elif dataset == 'vggface':
                        class_index = 1802
                    if str1 in img or str2 in img:
                        class_index = int(img.split('_')[0]) + class_index
                        save_dir_gen_moved = os.path.join(save_dir_gen, str(class_index))
                        os.makedirs(save_dir_gen_moved, exist_ok=True)
                        save_name = img
                        shutil.copyfile(generate_data + '/' +  img, save_dir_gen_moved + "\\" + "" + save_name)
    return

sobel_x = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
sobel_y = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

# def conv_operator(image, kernel=laplace, in_channels=3, out_channels=3):
#     output_laplace = F.conv2d(image, kernel.repeat(out_channels, in_channels, 1, 1).cuda(), stride=1, padding=1,)
#
#     return output_laplace

def npy_png(file_dir, dest_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # file =  + 'animal_128.npy'
    con_arr = np.load(file_dir)
    for clsidx in range(0, con_arr.shape[0]):
        os.makedirs(os.path.join(dest_dir, str(clsidx)), exist_ok=True)
        save_dir = os.path.join(dest_dir, str(clsidx))
        for imgs in range(0, con_arr.shape[1]):# 循环数组 最大值为图片张数（我的是200张）  三维数组分别是：图片张数  水平尺寸  垂直尺寸
                arr = con_arr[clsidx, imgs, :, :, :]  # 获得第i张的单一数组
                # disp_to_img = scipy.misc.imresize(arr, [375, 1242])  # 根据需要的尺寸进行修改
                plt.imsave(os.path.join(save_dir, "{}_disp_{}.png".format(clsidx,imgs)), arr, cmap='plasma')  # 定义命名规则，保存图片为彩色模式
                print('class_idx {} {} sample finished'.format(clsidx, imgs))

if __name__ == '__main__':
    # fileDir = "D:/Z-kobeshegu/Datasets/animal_png/test_original/"
    # trainDir = 'D:/Z-kobeshegu/Datasets/animal_png/'
    # moveFile(fileDir, trainDir, dataset='animal', generate_data='D:/Z-kobeshegu/Datasets/animal_png/test_waveGAN_M/train/136')

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', type=str, default="D:/Z-kobeshegu/Datasets/ACMMM2023_Classification/vggface_png_origin")
    parser.add_argument('--dest_dir', type=str, default="D:/Z-kobeshegu/Datasets/ACMMM2023_Classification/vggface_png_origin/")
    parser.add_argument('--generate_data', type=str, default="D:/Z-kobeshegu/Datasets/animal_png/test_waveGAN_M/train/136")
    parser.add_argument('--image_files', type=str, default="")
    args = parser.parse_args()
    # npy_png(file_dir=args.file_dir, dest_dir=args.dest_dir)
    moveFile(args.file_dir, args.dest_dir)

