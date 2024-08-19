import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np

import numpy as np
import torch
import math
from pathlib import Path
from torchvision import utils as vutils

def random_walk_mask(size):
    mask = np.zeros(size, dtype=np.int32)
    x, y = size[0] // 2, size[1] // 2  # Start in the center
    for _ in range(size[0] * size[1] // 2):  # Number of steps
        mask[x, y] = 1
        # Randomly move in one of the four directions
        direction = np.random.choice(['up', 'down', 'left', 'right'])
        if direction == 'up' and x > 0:
            x -= 1
        elif direction == 'down' and x < size[0] - 1:
            x += 1
        elif direction == 'left' and y > 0:
            y -= 1
        elif direction == 'right' and y < size[1] - 1:
            y += 1
    return mask


maska = torch.zeros(20, 3, 16, 16)
for i in range(20):
    # Generate the mask
    mask_size = (16, 16)
    mask = random_walk_mask(mask_size)
    # mask = torch.tensor(mask, dtype=torch.bool).view(-1).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.bool)
    mask = ~mask
    mask = mask.view(1, 16, 16)
    maska[i] = mask

mask_path = Path('./gaussian_mask.png')
vutils.save_image(maska, mask_path, nrow=5) 


maskb = torch.zeros(20, 3, 16, 16)
for i in range(20):
    r = math.floor(np.random.uniform() * 256)
    sample = torch.rand((1, 256)).topk(r, dim=1).indices
    mask = torch.zeros((1, 256), dtype=torch.bool)
    mask.scatter_(dim=1, index=sample, value=True)
    mask = mask.reshape((1, 16, 16))
    maskb[i] = mask

# print(mask.shape)
# plt.imshow(mask.numpy(), cmap='gray')
# plt.axis('off')
# plt.savefig('random_mask.png', bbox_inches='tight', pad_inches=0)

mask_path = Path('./random_mask.png')
vutils.save_image(maskb, mask_path, nrow=5) 