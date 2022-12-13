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
            nn.BatchNorm2d(num_features=nf_input, eps=1e-5, momentum=0.99),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=nf_input, out_channels=nf_output, kernel_size=kernel_size, padding='same',
                      bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=nf_output, eps=1e-5, momentum=0.99),
            nn.ReLU(),
            nn.Conv2d(in_channels=nf_output, out_channels=nf_output, kernel_size=kernel_size, padding='same'),
            nn.ReLU()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        residual = self.network(x)
        out = identity + residual
        return out


class Res_Block_down(nn.Module):
    def __init__(self, size, nf_input, nf_output, kernel_size=3):
        super(Res_Block_down, self).__init__()
        self.shortcut = Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=nf_input, out_channels=nf_output, kernel_size=1, padding='same'),
            nn.ReLU(),
        )
        self.network = Sequential(
            nn.LayerNorm(normalized_shape=[nf_input, size, size]),
            nn.ReLU(),
            nn.Conv2d(in_channels=nf_input, out_channels=nf_input, kernel_size=kernel_size, padding='same', bias=False),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=[nf_input, size, size]),
            nn.ReLU()
            nn.Conv2d(in_channels=nf_input, out_channels=nf_output, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        residual = self.network(x)
        out = identity + residual
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.nf = 64
        self.linear = nn.Linear(100, 4 * 4 * 8 * self.nf)
        self.res_block1 = Res_Block_up(self.nf * 8, self.nf * 4)
        self.res_block2 = Res_Block_up(self.nf * 4, self.nf * 2)
        self.res_block3 = Res_Block_up(self.nf * 2, self.nf * 1)
        self.bn = nn.BatchNorm2d(num_features=self.nf, eps=1e-5, momentum=0.99)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=self.nf, out_channels=3, kernel_size=3, padding='same')
        self.tanh = nn.Tanh()

        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 8 * self.nf, 4, 4)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.conv1(x)
        out = self.tanh(x)
        return out


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.nf = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.nf * 1, kernel_size=3, padding='same')
        self.res_block1 = Res_Block_down(size=32, nf_input=self.nf * 1, nf_output=self.nf * 2, kernel_size=3)
        self.res_block2 = Res_Block_down(size=16, nf_input=self.nf * 2, nf_output=self.nf * 4, kernel_size=3)
        self.res_block3 = Res_Block_down(size=8, nf_input=self.nf * 4, nf_output=self.nf * 8, kernel_size=3)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=512 * 4 * 4, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


generator = Generator()
critic = Critic()
test = torch.zeros((100, 100))
print(test.shape)
out = generator(test)
print(out.shape)
test2 = torch.zeros((100, 3, 32, 32))
res = critic(test2)
print(res.shape)
writer = SummaryWriter('./summary')
writer.add_graph(generator, test, verbose=False)
writer.add_graph(critic, test2, verbose=False)
writer.close()
