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
from utils import log

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
        log('Dataset has been created!')

    
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
    def __init__(self, data_dir, cls_='img', train=True, transform=None):
        self.cls_ = cls_
        self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
        self.std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
        self.transform = transform
        if train:
            self.files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
        else:
            self.files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
        log('Dataset has been created!')

                
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafile = self.files[index]
        with h5py.File(datafile, 'r') as hf:
            data = hf[self.cls_][:]
        image = np.array(data)

        if self.cls_ == 'img':
            for i in range(len(self.mean)):
                image[:,:, i] -= self.mean[i]
                image[:,:, i] /= self.std[i]

        image = image[:, :, 3:0:-1].copy()
        image[np.isnan(image)] = 0.000001

        if self.transform is not None:
            image = self.transform(image).float()  

        return image

class MNIST4Detect(Dataset):
    def __init__(self, root_dir, train, label, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.label = label
        if self.train:
            self.data_dir = os.path.join(self.root_dir, 'train', str(self.label))
        else:
            self.data_dir = os.path.join(self.root_dir, 'test', str(self.label))
        print(self.data_dir)
        self.file_list = sorted(glob.glob(os.path.join(self.data_dir, '*.png')))
        log('Dataset has been created!')

    def __getitem__(self, index):
        file_ = self.file_list[index]
        image = scipy.misc.imread(file_)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.file_list)

# if __name__ == '__main__':
#     train_transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     dataset = LandslideDataSet(data_dir='/exp/WGAN-GP-pytorch/data/landslide/img/no', cls_='img', train=True, transform=None)
#     print(dataset[0])
    

# with h5py.File('/exp/WGAN-GP-pytorch/data/landslide/img/no/image_3796.h5', 'r') as f:
#     image = f['img'][:]


# print(image.shape)