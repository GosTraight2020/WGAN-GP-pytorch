from load_dataset import Mnist32Dataset, LandslideDataSet
import torch 
from torch.utils.data import DataLoader
from model import WGAN_GP
from torchvision import transforms
from torch import optim, autograd
import torchvision
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from uitls import merge, log
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
    model.summary_writer.add_graph(model.generator, [torch.randn(128, 100), torch.randn(128, 10)])
    save_path = ('./eval/{}_{}/'.format(args.dataset, str(data_shape[1])))
    if not os.path.exists(save_path):
        log('Creating directory to save images: {}.'.format(save_path))
        os.makedirs(save_path, exist_ok=True)

    for epoch in range(args.num_epochs):
        for i, (real, label) in enumerate(data_loader):
            step = num_per_epoch * epoch + i
            for _ in range(5):
                noise = torch.randn(args.batch_size, 100)
                C_loss, W_dist, gp = model.train_critic_one_epoch(real, noise)
            
            G_loss, fake = wgan_gp.train_generator_one_epoch(real, noise)

            print('{}, G_loss : {} , C_loss :{}'.format(i, G_loss, C_loss))
            model.summary_writer.add_scalar('loss/C_loss', C_loss, global_step=step)
            model.summary_writer.add_scalar('loss/G_loss', G_loss, global_step=step)
            model.summary_writer.add_scalar('metric/W_dist', W_dist, global_step=step)
            model.summary_writer.add_scalar('metric/gradient_penalty', gp, global_step=step)
            #model.summary_writer.add_image('sample/fake', (fake[:64]+1)/2, global_step=step, dataformats='NCHW')
            #model.summary_writer.add_image('sample/real', (real[:64]+1)/2, global_step=step, dataformats='NCHW')
            model.summary_writer.flush()

            if step % 50 == 0:
                fake = fake.detach().numpy()
                fake = fake[:36]
                fake = fake.transpose(0, 3, 2, 1)
                save_img = merge(fake, [6, 6])
                img_save_path = os.path.join(save_path, 'step_{}.png'.format(step))
                log('Image of step {} has been save to {}'.format(step, img_save_path), level=3)
                scipy.misc.imsave(img_save_path, save_img)
        
            if step != 0 and step % args.ckpt_step == 0:
                torch.save(wgan_gp.generator, os.path.join(wgan_gp.checkpoint_path, 'generator_{}_step.pth'.format(step)))
                log('Model of step {} has been save to {}'.format(step, wgan_gp.checkpoint_path), level=3)





if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()
    data_shape = [3, 128, 128]

    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([-0.0625, -0.1277, -0.3074], [0.8869, 0.8860, 0.8775])
    ])

    assert args.dataset in ['mnist', 'landslide']
    log('Dataset is :{}'.format(args.dataset))

    if args.dataset == 'mnist':
        dataset = Mnist32Dataset(transform=train_transform)
        log('Dataset has been created!')
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
        log('DataLoader has been created!')
        wgan_gp = WGAN_GP(dataset=args.dataset, data_shape=data_shape, C_lr=1e-4, G_lr=1e-4, summary_path='./summary/', checkpoint_path='./checkpoints')        
        log('Model has been created!')

    if args.dataset == 'landslide':
        train_dataset = LandslideDataSet(data_dir='/exp/gan_gan_gan/data/landslide4scene/', list_path='/exp/gan_gan_gan/data/landslide4scene/TrainData/train.txt', transform=train_transform)
        log('Dataset has been created!')
        data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        log('DataLoader has been created!')
        wgan_gp = WGAN_GP(dataset=args.dataset, data_shape=data_shape, C_lr=1e-4, G_lr=1e-4, summary_path='./summary/', checkpoint_path='./checkpoints')        
        log('Model has been created!')

    num_per_epoch = len(data_loader)

    log('Going to train model........')
    train(wgan_gp, data_loader, args, num_per_epoch, data_shape)

    
 

    
