from load_dataset import Mnist32Dataset
import torch 
from torch.utils.data import DataLoader
from model import Generator, Critic
from torchvision import transforms
from torch import optim, autograd
import torchvision
from torch.utils.tensorboard import SummaryWriter

batch_size = 128
num_epochs = 30

def calc_gradient_penalty(critic, real_data, fake_data):
    batch_size, c, h, w = real_data.shape
    alpha = torch.rand([batch_size, 1, 1, 1]).repeat(1, c, h, w)
    interpolate = alpha * real_data + (1-alpha) * fake_data
    C_inter = critic(interpolate)

    grad = autograd.grad(
        outputs=C_inter, 
        inputs=interpolate, 
        grad_outputs=torch.ones_like(C_inter),
        create_graph=True, 
        retain_graph=True)[0]

    grad = grad.view(grad.size(0), -1)
    gp = ((grad.norm(2, dim=1)-1)**2).mean()
    return gp

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32))
])
dataset = Mnist32Dataset(transform=train_transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
generator = Generator()
critic = Critic()
num_per_epoch = len(data_loader)
one = torch.FloatTensor([1])
mone = one * -1

optimizerC = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
optimizerG = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))

writer = SummaryWriter('./summary', flush_secs=30)
for epoch in range(num_epochs):
    for i, (real_image, label) in enumerate(data_loader):
        step = num_per_epoch * epoch + i
        for _ in range(5):
            real = real_image
            noise = torch.randn(batch_size, 100)
            fake = generator(noise)
            C_real = critic(real)
            C_fake = critic(fake)
            gp = calc_gradient_penalty(critic, real, fake)
            W_dist = C_real.mean() - C_fake.mean()
            C_loss = -W_dist + 10*gp
            critic.zero_grad()
            C_loss.backward(retain_graph=True)
            optimizerC.step()

        noise = torch.randn(batch_size, 100)
        fake = generator(noise)
        C_fake = critic(fake)
        G_loss = -torch.mean(C_fake)
        generator.zero_grad()
        G_loss.backward()
        optimizerG.step()

        print('{}, G_loss : {} , C_loss :{}'.format(i, G_loss, C_loss))
        writer.add_scalar('C_loss', C_loss, global_step=step)
        writer.add_scalar('G_loss', G_loss, global_step=step)
        writer.add_image('generated', (fake[:64]+1)/2, global_step=step, dataformats='NCHW')
        writer.add_image('real', (real[:64]+1)/2, global_step=step, dataformats='NCHW')
        writer.flush()
    


    
