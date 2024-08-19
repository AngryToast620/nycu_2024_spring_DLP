import argparse
from pathlib import Path
from oxford_pet import load_dataset, OxfordPetDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from models import unet, resnet34_unet
from utils import dice_score, plot_compare_graph, save_images
from torchcontrib.optim import SWA

def new_image_mask(args, test_data):
    model_root = Path(args.model)
    batch_size = args.batch_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    torch.manual_seed(36)
    torch.cuda.manual_seed(36)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False)
    
    # print(args.model.split('/')[2].split('_')[1])
    model1 = unet.UNet(input_shape=3,
                       output_shape=1).to(device)
    model2 = resnet34_unet.ResNet_UNet().to(device)

    # swa_model1 = torch.optim.swa_utils.AveragedModel(model1)
    # swa_model2 = torch.optim.swa_utils.AveragedModel(model2)

    model1_path = model_root / 'unet_best' / 'unet_best.pth'
    model2_path = model_root / 'res_unet_best' / 'res_unet_best.pth'

    model1.load_state_dict(torch.load(model1_path))
    model2.load_state_dict(torch.load(model2_path))

    # swa_model1.load_state_dict(torch.load(model1_path))
    # swa_model2.load_state_dict(torch.load(model2_path))
    
    unet_acc = 0
    res_unet_acc = 0
    image = []
    unet_mask = []
    res_unet_mask = []

    model1.eval()
    model2.eval()

    # swa_model1.eval()
    # swa_model2.eval()
    with torch.inference_mode():
        for batch, (images, masks, trimaps) in enumerate(test_dataloader):
            images, masks = images.to(device), masks.to(device)
            image.append([images, masks])

            # ---------- UNet ----------
            outputs = model1(images)
            # outputs = swa_model1(images)

            output_logit = torch.sigmoid(outputs)
            unet_mask.append(output_logit)
            unet_acc += dice_score(torch.round(output_logit), masks)

            # ---------- ResNet UNet ----------
            outputs = model2(images)
            # outputs = swa_model2(images)

            output_logit = torch.sigmoid(outputs)
            res_unet_mask.append(output_logit)
            res_unet_acc += dice_score(torch.round(output_logit), masks)


        unet_acc = unet_acc / len(test_dataloader)
        res_unet_acc = res_unet_acc / len(test_dataloader)

        plot_compare_graph(model1, model2, image, n=3)

        # save_dir = Path('./masks/')
        # save_images(folder_path=save_dir/'UNet', photo_list=unet_mask)
        # save_images(folder_path=save_dir/'ResNet34 UNet', photo_list=res_unet_mask)
    
    return unet_acc, res_unet_acc


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    data_root = Path(args.data_path)
    data_path = load_dataset(data_path=data_root)

    def test_transform(image, mask, trimap):
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        image_transform = transforms.Compose([
            transforms.Normalize(mean=[0.478, 0.446, 0.396],
                                 std=[0.226, 0.223, 0.225])
        ])

        # transformed_sample = [data_transform(image), data_transform(mask), data_transform(trimap)]
        transformed_sample = [image_transform(data_transform(image)), data_transform(mask), data_transform(trimap)]
        return transformed_sample

    test_data = OxfordPetDataset(root=data_path,
                                 mode="test",
                                 transform=test_transform)
    
    unet_acc, res_unet_acc = new_image_mask(args, test_data)

    print(f'UNet dice score: {unet_acc:.4f} | ResNet34 UNet dice scroe: {res_unet_acc:.4f}')