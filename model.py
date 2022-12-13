import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter

class Res_Block_up(nn.Module):
    def __init__(self, nf_input, nf_output, kernel_size=3):
        super(Res_Block_up, self).__init__()
        self.shortcut = Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=nf_input, out_channels=nf_output, kernel_size=1, padding='same')
        )
        self.network = Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(num_features=nf_input, eps=1e-5, momentum=0.99),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=nf_input, out_channels=nf_output, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=nf_output, eps=1e-5, momentum=0.99),
            nn.ReLU(),
            nn.Conv2d(in_channels=nf_output, out_channels=nf_output, kernel_size=kernel_size, padding='same'),
            nn.ReLU()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.network(x)
        out = out+residual
        return out

class Res_Block_down(nn.Module):
    def __init__(self, nf_input, nf_output, kernel_size=3):
        super(Res_Block_down, self).__init__()
        self.shortcut = Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=nf_inputm out_channels=nf_output, kernel_size=1, padding='same')
            
        )
        self.network = Sequential(

        )

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.nf = 64
        self.linear = nn.Linear(100, 4*4*8*self.nf)
        self.res_block1 = Res_Block_up(self.nf*8, self.nf*4)
        self.res_block2 = Res_Block_up(self.nf*4, self.nf*2)
        self.res_block3 = Res_Block_up(self.nf*2, self.nf*1)
        self.bn = nn.BatchNorm2d(num_features=self.nf, eps=1e-5, momentum=0.99)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=self.nf, out_channels=3, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()

        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding='same')
    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 8*self.nf, 4, 4)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.conv1(x)
        out = self.relu2(x)
        return out

generator = Generator()
test = torch.zeros((100, 100))
print(test.shape)
out = generator(test)
print(out.shape)
writer = SummaryWriter('./summary')
writer.add_graph(generator, test, verbose=False)
writer.close()