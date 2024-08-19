import torch
from pathlib import Path
from torchvision import transforms
from oxford_pet import OxfordPetDataset
import matplotlib.pyplot as plt
import random
# from models import unet, resnet34_unet
# from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt


def dice_score(pred_mask, gt_mask):
    pred_mask = pred_mask.view(pred_mask.size(0), -1)
    gt_mask = gt_mask.view(gt_mask.size(0), -1)

    intersection = torch.sum(pred_mask == gt_mask).item()

    total_size = pred_mask.size(0) * pred_mask.size(1) + gt_mask.size(0) * gt_mask.size(1)

    score = (2 * intersection) / total_size

    return score


def mean_std():
    path = Path("../dataset/oxford-iiit-pet/")

    def train_transform(image, mask, trimap):
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        transformed_sample = [data_transform(
            image), data_transform(mask), data_transform(trimap)]
        return transformed_sample

    train_data = OxfordPetDataset(root=path,
                                  mode='train',
                                  transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)

    for (images, masks, trimaps) in train_loader:
        for d in range(3):
            mean[d] += images[:, d, :, :].mean()
            std[d] += images[:, d, :, :].std()

    mean.div_(len(train_loader))
    std.div_(len(train_loader))
    print(list(mean.numpy()), list(std.numpy()))

# mean_std()



def plot_compare_graph(model1, model2, data, n=3):
    random_image = random.sample(data, k=n)

    model1.eval()
    model2.eval()

    for i, (img, mask) in enumerate(random_image):
        fig, ax = plt.subplots(1, 4)

        origin_img = img.squeeze(0)
        origin_img = origin_img.permute(1, 2, 0)
        ax[0].imshow(origin_img.cpu())
        ax[0].set_title(f"Original img")
        ax[0].axis("off")

        # Transform and plot image
        # Note: permute() will change shape of image to suit matplotlib
        # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])

        u_mask = model1(img)
        u_mask = torch.sigmoid(u_mask).squeeze().cpu()
        u_mask = u_mask.numpy()
        u_mask = Image.fromarray((u_mask*255).astype(np.uint8), mode='L')
        # u_mask = Image.fromarray(u_mask, mode='L')
        ax[1].imshow(u_mask, cmap='gray')
        ax[1].set_title(f"UNet")
        ax[1].axis("off")

        r_mask = model2(img)
        r_mask = torch.sigmoid(r_mask).squeeze().cpu()
        r_mask = r_mask.numpy()
        r_mask = Image.fromarray((r_mask*255).astype(np.uint8), mode='L')
        # r_mask = Image.fromarray(r_mask, mode='L')
        ax[2].imshow(r_mask, cmap='gray')
        ax[2].set_title(f"ResNet34-UNet")
        ax[2].axis("off")

        mask = mask.squeeze(0)
        mask = mask.permute(1, 2, 0)
        ax[3].imshow(mask.cpu())
        ax[3].set_title(f"GT mask")
        ax[3].axis("off")

        fig.suptitle(f"Compare graph {i+1}", fontsize=16)
        plt.pause(1)  # stop 1 second
        plt.show()
    # input("Press Enter to close...")



def save_images(folder_path, photo_list):
    for idx, tensor_img in enumerate(photo_list):
        photo_path = os.path.join(folder_path, f"mask_{idx}.jpg")
        try:
            tensor_img = tensor_img.squeeze().cpu()
            img_array = tensor_img.numpy()  
            img = Image.fromarray((img_array*255).astype(np.uint8), mode='L')
            img.save(photo_path)
            print(f"Photo saved successfully: {photo_path}")
        except Exception as e:
            print(f"Error saving photo: {e}")
            break

def acc_plot(v_train, v_valid, r_train, r_valid):
    epoch = np.arange(1, 31)
    plt.plot(epoch, v_train, label='VGG19_train_acc', color='b')
    plt.plot(epoch, v_valid, label='VGG19_valid_acc', color='y')
    plt.plot(epoch, r_train, label='ResNet50_train_acc', color='g')
    plt.plot(epoch, r_valid, label='ResNet50_valid_acc', color='r')

    plt.legend()
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(False)

    plt.show()

# u_result_dir = Path('../test_saved_models/model_UNet_5/result.pkl')
# r_result_dir = Path('../test_saved_models/model_Res_UNet_4/result.pkl')

# with open(u_result_dir, 'rb') as f:
#     u_result = pickle.load(f)

# with open(r_result_dir, 'rb') as f:
#     r_result = pickle.load(f)

# u_train_acc = [i*100 for i in u_result['train acc']]
# r_train_acc = [i*100 for i in r_result['train acc']]

# u_valid_acc = [i*100 for i in u_result['valid acc']]
# r_valid_acc = [i*100 for i in r_result['valid acc']]

# acc_plot(u_train_acc, u_valid_acc, r_train_acc, r_valid_acc)