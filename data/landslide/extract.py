import numpy as np
import glob
import os 
import h5py
import scipy.misc
import shutil 

data_dir = './data/landslide/'
img_dir = os.path.join(data_dir, 'image')
mask_dir = os.path.join(data_dir, 'mask')
image_dir_yes = os.path.join(img_dir, 'yes')
image_dir_no = os.path.join(img_dir, 'no')
mask_dir_yes = os.path.join(mask_dir, 'yes')
mask_dir_no = os.path.join(mask_dir, 'no')

for d in [image_dir_yes, image_dir_no, mask_dir_yes, mask_dir_no]:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

file_pattern = os.path.join(img_dir, '*.h5')
file_list = sorted(glob.glob(file_pattern))

num = 0
for img_file in file_list:
    mask_file = img_file.replace('img', 'mask').replace('image', 'mask')
    img_name = img_file.split('/')[-1]
    mask_name = mask_file.split('/')[-1]
    print(mask_file)
    with h5py.File(mask_file, 'r') as f:
        mask = f['mask']
        mask = np.array(mask)
        # non-landslide
        if mask[mask==1].size == 0:
            img_from = img_file
            img_to = os.path.join(image_dir_no, img_name)
            mask_from = mask_file
            mask_to = os.path.join(mask_dir_no, mask_name)
            print('{}, img_from: {}, img_to:{}, mask_from:{}, mask_to:{}'.format(mask[mask==1].size, img_from, img_to, mask_from, mask_to))
        else:
            img_from = img_file
            img_to = os.path.join(image_dir_yes, img_name)
            mask_from = mask_file
            mask_to = os.path.join(mask_dir_yes, mask_name)
            print('{}, img_from: {}, img_to:{}, mask_from:{}, mask_to:{}'.format(mask[mask==1].size, img_from, img_to, mask_from, mask_to))
        shutil.move(img_from, img_to)
        shutil.move(mask_from, mask_to)

