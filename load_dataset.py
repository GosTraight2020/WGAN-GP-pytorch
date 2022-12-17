# import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy
import numpy as np
import os, glob
import json
import scipy.misc
import PIL.Image as Image
import torch.nn.functional as F
import h5py
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
        label = torch.from_numpy(np.array(int(label)))
        label = F.one_hot(input=label, num_classes=10)
        label = label.float()
        return img, label

    def __len__(self):
        return len(self.img_file_list)

class LandslideDataSet(Dataset):
    def __init__(self, data_dir, list_path, max_iters=None,set='unlabeled', transform=None):
        self.list_path = list_path
        self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
        self.std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.transform = transform
        
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []

        if set=='labeled':
            for name in self.img_ids:
                img_file = data_dir + name
                label_file = data_dir + name.replace('img','mask').replace('image','mask')
                self.files.append({
                    'img': img_file,
                    'label': label_file,
                    'name': name
                })
        elif set=='unlabeled':
            for name in self.img_ids:
                img_file = data_dir + name
                self.files.append({
                    'img': img_file,
                    'name': name
                })
                
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        with h5py.File(datafiles['img'], 'r') as hf:
            image = hf['img'][:]  
        name = datafiles['name'] 
        image = np.asarray(image, np.float32)
        size = image.shap
        image = image[:, :, 3:0:-1].copy()
        image[np.isnan(image)] = 0.000001

        if self.transform is not None:
            image = self.transform(image).float()  

        return image, torch.ones(1).float()





