from load_dataset import MNIST4Detect, LandslideDataSet
import torch 
from torch.utils.data import DataLoader
from model import WGAN_GP, AutoEncoder
from torchvision import transforms
from torch import optim, autograd
import torchvision
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from utils import merge, log
import os
import scipy.misc

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', default=20, help='default: 20', type=int)
    parser.add_argument('--batch_size', default=128, help='default: 128', type=int)
    parser.add_argument('--dataset', '-D', help='mnist', required=True)
    parser.add_argument('--ckpt_step', default=100, help='# of steps for saving checkpoint (default: 5000)', type=int)
    return parser

def train(model, data_loader, args, num_per_epoch, data_shape):
    model.summary_writer.add_graph(model.ae, torch.randn(128, data_shape[0], data_shape[1], data_shape[2]))
    # if not os.path.exists(save_path):
    #     log('Creating directory to save images: {}.'.format(save_path))
    #     os.makedirs(save_path, exist_ok=True)

    for epoch in range(args.num_epochs):
        for i, real in enumerate(data_loader):
            step = num_per_epoch * epoch + i
            for j in range(5):
                C_loss, W_dist, gp = model.train_critic_one_epoch(real)
            G_loss, recon_loss, loss, fake = model.train_generator_one_epoch(real)
            # recon_loss, fake = model.train_generator_one_epoch(real)
            print('{}, l1_loss : {}'.format(step, recon_loss))
            model.summary_writer.add_scalar('loss/C_loss', C_loss, global_step=step)
            model.summary_writer.add_scalar('loss/G_loss', G_loss, global_step=step)
            model.summary_writer.add_scalar('loss/recon_loss', recon_loss, global_step=step)
            model.summary_writer.add_scalar('loss/loss', loss, global_step=step)
            model.summary_writer.add_scalar('metric/W_dist', W_dist, global_step=step)
            model.summary_writer.add_scalar('metric/gradient_penalty', gp, global_step=step)
            # model.summary_writer.add_image('sample/fake', (fake[:64]+1)/2, global_step=step, dataformats='NCHW')
            # model.summary_writer.add_image('sample/real', (real[:64]+1)/2, global_step=step, dataformats='NCHW')
            model.summary_writer.flush()

            if step % 50 == 0:
                fake = fake.detach().numpy().transpose(0, 3, 2, 1)
                fake = fake[:36]
                real = real.detach().numpy().transpose(0, 3, 2, 1)
                real = real[:36]
                save_fake_img = merge(fake, [6, 6])
                save_real_img = merge(real, [6, 6])
                img_fake_path = os.path.join(model.eval_path, 'step_{}_fake.png'.format(step))
                img_real_path = os.path.join(model.eval_path, 'step_{}_real.png'.format(step))
                log('Image of step {} has been save to {}'.format(step, img_fake_path), level=3)
                log('Image of step {} has been save to {}'.format(step, img_real_path), level=3)
                scipy.misc.imsave(img_fake_path, save_fake_img)
                scipy.misc.imsave(img_real_path, save_real_img)
        
            if step != 0 and step % args.ckpt_step == 0:
                model_save_path = os.path.join(model.checkpoint_path, 'generator_{}_step.pth'.format(step))
                torch.save(model.ae, model_save_path)
                log('Model of step {} has been save to {}'.format(step, model_save_path), level=3)

if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()
    data_shape = [3, 128, 128]
    assert args.dataset in ['mnist', 'landslide']
    log('Dataset is :{}'.format(args.dataset))

    if args.dataset == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize([0.5], [0.5])
          ])
        dataset = MNIST4Detect(root_dir='./data/mnist/', train=True, label=0, transform=train_transform)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
        ae = AutoEncoder(args.dataset, data_shape, C_lr=1e-4, G_lr=1e-4, summary_path='./summary/', checkpoint_path='./checkpoints')
    elif args.dataset == 'landslide':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
          ])
        dataset = LandslideDataSet(data_dir='/exp/WGAN-GP-pytorch/data/landslide/img/no', cls_='img', train=True, transform=train_transform)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        ae = AutoEncoder(args.dataset, data_shape, C_lr=1e-4, G_lr=1e-4, summary_path='./summary/', checkpoint_path='./checkpoints')

    num_per_epoch = len(data_loader)

    train(ae, data_loader, args, num_per_epoch, data_shape)
    
 

    
