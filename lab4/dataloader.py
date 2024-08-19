
import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData

from torchvision.datasets.folder import default_loader as imgloader
from torch import stack

# import torchvision.utils as vutils
# from pathlib import Path

# def save_images(tensor, directory, i):
#     # Create the directory if it doesn't exist
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     # Iterate over each image in the batch
#     for j in range(tensor.size(0)):
#         # Create the file path for the image
#         image_path = os.path.join(directory, f'image_{i}.png')

#         # Save the image
#         vutils.save_image(tensor[j], image_path)

def get_key(fp):
    filename = fp.split('\\')[-1]
    filename = filename.split('.')[0].replace('frame', '')
    return int(filename)

class Dataset_Dance(torchData):
    """
        Args:
            root (str)      : The path of your Dataset
            transform       : Transformation to your dataset
            mode (str)      : train, val, test
            partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """
    def __init__(self, root, transform, mode='train', video_len=7, partial=1.0):
        super().__init__()
        assert mode in ['train', 'val'], "There is no such mode !!!"
        if mode == 'train':
            self.img_folder     = sorted(glob(os.path.join(root, 'train\\train_img\\*.png')), key=get_key)
            self.prefix = 'train'
        elif mode == 'val':
            self.img_folder     = sorted(glob(os.path.join(root, 'val\\val_img\\*.png')), key=get_key)
            self.prefix = 'val'
        else:
            raise NotImplementedError
        
        self.transform = transform
        self.partial = partial
        self.video_len = video_len

    def __len__(self):
        return int(len(self.img_folder) * self.partial) // self.video_len

    def __getitem__(self, index):
        path = self.img_folder[index]
        
        imgs = []
        labels = []
        for i in range(self.video_len):
            label_list = self.img_folder[(index*self.video_len)+i].split('\\')
            label_list[-2] = self.prefix + '_label'
            
            img_name    = self.img_folder[(index*self.video_len)+i]
            label_name = '\\'.join(label_list)

            imgs.append(self.transform(imgloader(img_name)))
            labels.append(self.transform(imgloader(label_name)))

        # for j in range(len(imgs)):
        #     save_images(labels[j], Path('./save_test_result/Cyclical'), j)

        return stack(imgs), stack(labels)
