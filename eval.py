from utils import merge, make_condition_label
import numpy as np
import torch
import imageio
import glob
import os

ckpt_dir = './checkpoints/mnist_32'
eval_dir = ckpt_dir.replace('checkpoints', 'eval')
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir, exist_ok=True)
ckpt_pattern = os.path.join(ckpt_dir, '*.pth')
ckpt_list = glob.glob(ckpt_pattern)
ckpt_list.sort()
label = make_condition_label()
noise = torch.randn(100, 100)

for ckpt in ckpt_list:
    step = ckpt.split('/')[-1].split('.')[0].split('_')[1]
    model = torch.load(ckpt)
    res = model(noise, label).view(-1, 32 ,32, 1)
    res = (res+1)/2
    res = np.array(res.tolist())
    img = merge(res, [10, 10])
    imageio.imsave(os.path.join(eval_dir, 'eval_{}.png'.format(step)), img)
    del model