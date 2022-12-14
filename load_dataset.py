# import torchvision
import imageio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy
import numpy as np
import os, glob
import json
import scipy.misc
import PIL.Image as Image
# train_data =  torchvision.datasets.MNIST(
#     root = './data/MINIST',  #数据集的位置
#     train = True,       #如果为True则为训练集，如果为False则为测试集
#     transform = torchvision.transforms.ToTensor(),   #将图片转化成取值[0,1]的Tensor用于网络处理
#     download=True
# )

class Mnist32Dataset(Dataset):
    def __init__(self, transform=None):
        self.root_dir = '/exp/gan_gan_gan/data/mnist'
        self.transform = transform
        self.img_pattern = os.path.join(self.root_dir, '*.png')
        self.label_file = os.path.join(self.root_dir, 'labels.json')
        self.img_file_list = glob.glob(self.img_pattern)
        self.img_file_list.sort()
        with open(self.label_file, 'r') as f:
            self.label_dict = json.load(f)
    
    def __getitem__(self, index):
        file_name = self.img_file_list[index]
        file_index = file_name.split('/')[-1].split('.')[0].split('_')[1]
        # img = scipy.misc.imread(file_name,)
        img = Image.open(file_name)
        img = np.array(img).astype('float32')
        img = img/127.5-1.0
        if self.transform is not None:
            img = self.transform(img)
        label = self.label_dict[file_index]
        return img, label

    def __len__(self):
        return len(self.img_file_list)

# dataset = Mnist32Dataset()
# data_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=1, drop_last=True)
# print(len(dataset))
# print(len(data_loader))







