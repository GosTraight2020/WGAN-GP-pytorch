import torchvision

train_data =  torchvision.datasets.MNIST(
    root = './data/MINIST',  #数据集的位置
    train = True,       #如果为True则为训练集，如果为False则为测试集
    transform = torchvision.transforms.ToTensor(),   #将图片转化成取值[0,1]的Tensor用于网络处理
    download=True
)

img, lable = train_data[0]
print(lable)