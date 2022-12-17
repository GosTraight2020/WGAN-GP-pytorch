import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter
from torch import optim, autograd
import os 

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
            nn.ReLU(),
            nn.Conv2d(in_channels=nf_input, out_channels=nf_output, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        residual = self.network(x)
        out = identity + residual
        return out

class Generator_32(nn.Module):
    def __init__(self, num_channels, nf):
        super(Generator_32, self).__init__()
        self.nf = nf
        self.num_channels = num_channels
        self.linear = nn.Linear(100, 4 * 4 * 8 * self.nf)
        self.res_block1 = Res_Block_up(self.nf * 8, self.nf * 4)
        self.res_block2 = Res_Block_up(self.nf * 4, self.nf * 2)
        self.res_block3 = Res_Block_up(self.nf * 2, self.nf * 1)
        self.bn = nn.BatchNorm2d(num_features=self.nf*1, eps=1e-5, momentum=0.99)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=self.nf*1, out_channels=self.num_channels, kernel_size=3, padding='same')
        self.tanh = nn.Tanh()

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

class Critic_32(nn.Module):
    def __init__(self, num_channels, nf):
        super(Critic_32, self).__init__()
        self.nf = nf
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.nf * 1, kernel_size=3, padding='same')
        self.res_block1 = Res_Block_down(size=32, nf_input=self.nf * 1, nf_output=self.nf * 2)
        self.res_block2 = Res_Block_down(size=16, nf_input=self.nf * 2, nf_output=self.nf * 4)
        self.res_block3 = Res_Block_down(size=8, nf_input=self.nf * 4, nf_output=self.nf * 8)
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

class Critic_128(nn.Module):
    def __init__(self, num_channels, nf):
        super(Critic_128, self).__init__()
        self.nf = nf
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.nf * 1, kernel_size=3, padding='same')
        self.res_block1 = Res_Block_down(size=128, nf_input=self.nf * 1, nf_output=self.nf * 2)
        self.res_block2 = Res_Block_down(size=64, nf_input=self.nf * 2, nf_output=self.nf * 4)
        self.res_block3 = Res_Block_down(size=32, nf_input=self.nf * 4, nf_output=self.nf * 8)
        self.res_block4 = Res_Block_down(size=16, nf_input=self.nf * 8, nf_output=self.nf * 16)
        self.res_block5 = Res_Block_down(size=8, nf_input=self.nf * 16, nf_output=self.nf * 16)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=self.nf * 16 * 4 * 4, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

class Generator_128(nn.Module):
    def __init__(self, num_channels, nf, activation='tanh'):
        super(Generator_128, self).__init__()
        self.nf = nf
        self.num_channels = num_channels
        self.linear = nn.Linear(in_features=100, out_features=4*4*16*self.nf)
        self.res_block1 = Res_Block_up(self.nf * 16, self.nf * 16)
        self.res_block2 = Res_Block_up(self.nf * 16, self.nf * 8)
        self.res_block3 = Res_Block_up(self.nf * 8, self.nf * 4)
        self.res_block4 = Res_Block_up(self.nf * 4, self.nf * 2)
        self.res_block5 = Res_Block_up(self.nf * 2, self.nf * 1)
        self.bn = nn.BatchNorm2d(num_features=self.nf*1, eps=1e-5, momentum=0.99)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=self.nf*1, out_channels=self.num_channels, kernel_size=3, padding='same')
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 16 * self.nf, 4, 4)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.conv1(x)
        out = self.activation(x)
        return out

class WGAN_GP:
    def __init__(self, dataset, data_shape, C_lr=1e-4, G_lr=1e-4, summary_path='./summary/', checkpoint_path='./checkpoints'):
        self.data_shape = data_shape
        self.C_lr = C_lr
        self.G_lr = G_lr
        self.lamda = 10.
        if self.data_shape[1] == 32:
            self.critic = Critic_32(num_channels=self.data_shape[0], nf=64)
            self.generator = Generator_32(num_channels=self.data_shape[0], nf=64)
        elif self.data_shape[2] == 128:
            self.critic = Critic_128(num_channels=self.data_shape[0], nf=32)
            self.generator = Generator_128(num_channels=self.data_shape[0], nf=32, activation='relu')
        self.C_opt = optim.Adam(self.critic.parameters(), lr=self.C_lr, betas=(0.0, 0.9))
        self.G_opt = optim.Adam(self.generator.parameters(), lr=self.G_lr, betas=(0.0, 0.9))
        self.summary_path = os.path.join(summary_path, dataset+'_'+str(self.data_shape[1]))
        self.checkpoint_path = os.path.join(checkpoint_path, dataset+'_'+str(self.data_shape[1]))
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path, exist_ok=True)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path, exist_ok=True)
        self.summary_writer = SummaryWriter(self.summary_path, flush_secs=30)

    def calc_gradient_penalty(self, real_data, fake_data):
        batch_size, c, h, w = real_data.shape
        alpha = torch.rand([batch_size, 1, 1, 1]).repeat(1, c, h, w)
        interpolate = alpha * real_data + (1-alpha) * fake_data
        C_inter = self.critic(interpolate)

        grad = autograd.grad(
            outputs=C_inter, 
            inputs=interpolate, 
            grad_outputs=torch.ones_like(C_inter),
            create_graph=True, 
            retain_graph=True)[0]

        grad = grad.view(grad.size(0), -1)
        gp = ((grad.norm(2, dim=1)-1)**2).mean()
        return gp
    
    def train_critic_one_epoch(self, real, noise):
        fake = self.generator(noise)
        C_real = self.critic(real)
        C_fake = self.critic(fake)
        gp = self.calc_gradient_penalty(real, fake)
        W_dist = C_real.mean() - C_fake.mean()
        C_loss = -W_dist + self.lamda*gp
        self.critic.zero_grad()
        C_loss.backward(retain_graph=True)
        self.C_opt.step()
        return C_loss, W_dist, gp

    def train_generator_one_epoch(self, real, noise):
        fake = self.generator(noise)
        C_fake = self.critic(fake)
        G_loss = -torch.mean(C_fake)
        self.generator.zero_grad()
        G_loss.backward()
        self.G_opt.step()
        return G_loss, fake

