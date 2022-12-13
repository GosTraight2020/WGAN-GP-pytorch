import torchvision
import imageio
import scipy
import numpy as np
train_data =  torchvision.datasets.MNIST(
    root = './data/MINIST',  #数据集的位置
    train = True,       #如果为True则为训练集，如果为False则为测试集
    transform = torchvision.transforms.ToTensor(),   #将图片转化成取值[0,1]的Tensor用于网络处理
    download=True
)

for i, data in enumerate(train_data):
    image, target = data
    image = np.array(image, dtype=np.uint8).reshape(28, 28)
    image = image/255.0
    print(image.dtype)
    imageio.imsave('./data/{}.png'.format(i), image)
