from utils import merge, make_condition_label, log
from argparse import ArgumentParser
import numpy as np
import torch
import scipy.misc
import glob
import os


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset', '-D', help='mnist', required=True)
    parser.add_argument('--number', help='number of samples', default=36, type=int)
    parser.add_argument('--size', help='size of each sample', required=True, type=int)
    parser.add_argument('--condition', action='store_true')

    return parser


def eval(args, N):
    ckpt_dir = os.path.join('./checkpoints', args.dataset+'_'+str(args.size))
    eval_dir = ckpt_dir.replace('checkpoints', 'eval')
    ckpt_pattern = os.path.join(ckpt_dir, '*.pth')
    ckpt_list = glob.glob(ckpt_pattern)

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=True)

    assert len(ckpt_list) != 0, 'Checkpoint files do not exist!'
    
    ckpt_list.sort()
    label = make_condition_label([10, 10])
    noise = torch.randn(args.number, 100)

    for ckpt in ckpt_list:
        step = ckpt.split('/')[-1].split('.')[0].split('_')[1]
        model = torch.load(ckpt)
        res = model(noise, label)
        res = res.detach().numpy().transpose(0, 3, 2, 1)
        img = merge(res, [int(N), int(N)])
        log('Save image of {} step to directory : {}'.format(step, eval_dir))
        scipy.misc.imsave(os.path.join(eval_dir, 'eval_{}.png'.format(step)), img)
        del model

if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    assert args.dataset in ['mnist', 'landslide'], 'Dataset {} not in default datasets list.'.format(args.dataset)
    assert args.size in [32, 128], 'Size {} not in default size list.'.format(args.size)
    
    N = args.number ** 0.5
    assert N == int(N),  'Number should be a square number'

    eval(args, N)
