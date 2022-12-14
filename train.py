from load_dataset import Mnist32Dataset
import torch 
from torch.utils.data import DataLoader
from model import WGAN_GP
from torchvision import transforms
from torch import optim, autograd
import torchvision
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import os

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', default=20, help='default: 20', type=int)
    parser.add_argument('--batch_size', default=128, help='default: 128', type=int)
    parser.add_argument('--dataset', '-D', help='mnist', required=True)
    parser.add_argument('--ckpt_step', default=100, help='# of steps for saving checkpoint (default: 5000)', type=int)
    parser.add_argument('--condition', action='store_true')

    return parser

def train(model, data_loader, args, num_per_epoch):
    for epoch in range(args.num_epochs):
        for i, (real, label) in enumerate(data_loader):
            step = num_per_epoch * epoch + i
            for _ in range(5):
                noise = torch.randn(args.batch_size, 100)
                C_loss, W_dist, gp = model.train_critic_one_epoch(real, label, noise)
            G_loss, fake = wgan_gp.train_generator_one_epoch(real, label, noise)

            print('{}, G_loss : {} , C_loss :{}'.format(i, G_loss, C_loss))
            model.summary_writer.add_scalar('loss/C_loss', C_loss, global_step=step)
            model.summary_writer.add_scalar('loss/G_loss', G_loss, global_step=step)
            model.summary_writer.add_scalar('metric/W_dist', W_dist, global_step=step)
            model.summary_writer.add_scalar('metric/gradient_penalty', gp, global_step=step)
            model.summary_writer.add_image('sample/fake', (fake[:64]+1)/2, global_step=step, dataformats='NCHW')
            model.summary_writer.add_image('sample/real', (real[:64]+1)/2, global_step=step, dataformats='NCHW')
            model.summary_writer.flush()
        
            if step % args.ckpt_step == 0:
                torch.save(wgan_gp.generator, os.path.join(wgan_gp.checkpoint_path, 'generator_{}_step.pth'.format(step)))
                print('------------generator model of {} step has been saved'.format(step))


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32))
])


if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()
    data_shape = [1, 32, 32]

    assert args.dataset in ['mnist']

    if args.dataset == 'mnist':
        dataset = Mnist32Dataset(transform=train_transform)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)

    num_per_epoch = len(data_loader)

    wgan_gp = WGAN_GP(dataset=args.dataset, data_shape=data_shape, C_lr=1e-4, D_lr=1e-4, summary_path='./summary/', checkpoint_path='./checkpoints')        
    train(wgan_gp, data_loader, args, num_per_epoch)

    
 

    
