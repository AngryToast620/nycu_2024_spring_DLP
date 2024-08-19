import pandas as pd
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

def getData(mode):
    if mode == 'train':
        path_to_train = './dataset/train.csv'
        df = pd.read_csv(path_to_train)
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    
    if mode == 'valid':
        path_to_valid = './dataset/valid.csv'
        df = pd.read_csv(path_to_valid)
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    
    else:
        path_to_test = './dataset/test.csv'
        df = pd.read_csv(path_to_test)
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label

class BufferflyMothLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))  

        # Write a transform for image
        self.train_transform = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(90),
            # Turn the image into a torch.Tensor
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """

        image_path = self.root + self.img_name[index]
        raw_img = Image.open(image_path)
        if self.mode == 'train':
            img = self.train_transform(raw_img)
        else:
            img = self.test_transform(raw_img)

        label = self.label[index]
        # label = torch.tensor(label)
        # print(type(img), type(label))
        return img, label
    
# root = './dataset/'
# bm = BufferflyMothLoader(root=root, mode='train')
# bm.__getitem__(1)