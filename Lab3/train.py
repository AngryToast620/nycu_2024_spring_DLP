import argparse
from models.unet import UNet
from models.resnet34_unet import ResNet_UNet
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from torchcontrib.optim import SWA
from oxford_pet import OxfordPetDataset, SimpleOxfordPetDataset, download_url, extract_archive, load_dataset
import utils
import evaluate
from pathlib import Path
from tqdm.auto import tqdm
import os 
import pickle

def train(args, train_data, valid_data, net, save_dir):
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    torch.manual_seed(36)
    torch.cuda.manual_seed(36)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_data,
                                  batch_size=batch_size,
                                  shuffle=False)

    if net == 'UNet':
        model = UNet(input_shape=3,
                     output_shape=1).to(device)
    else:
        model = ResNet_UNet().to(device)
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    swa_scheduler = SWA(optimizer, swa_start=int(0.8*epochs), swa_freq=5)
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    
    results = {'train acc': [],
               'train loss': [],
               'valid acc': []}

    # save_dir = Path(f"../saved_models/model_{net}_4")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(epochs)):
        train_acc, train_loss = 0, 0
        valid_acc = 0
        model.train()
        for batch, (images, masks, trimaps) in enumerate(train_dataloader):
            
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)

            loss = loss_fn(outputs, masks)
            train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            # scheduler.step()

            output_logit = torch.sigmoid(outputs)
            # output_pred = torch.round(output_logit)
            # train_acc += (output_pred == masks).float().mean().item()
            
            train_acc += utils.dice_score(torch.round(output_logit), masks)


        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)

        valid_acc = evaluate.evaluate(model, valid_dataloader, device)

        print(f'Epoch: {epoch+1} | train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | valid acc: {valid_acc:.4f}')

        results['train acc'].append(train_acc)
        results['train loss'].append(train_loss)
        results['valid acc'].append(valid_acc)

        if epoch >= int(0.8 * epochs):
            swa_model.update_parameters(model)
            swa_scheduler.step()
        if epoch >= epochs-5:
            model_path = save_dir / f"epoch{epoch}_acc{valid_acc:.3f}.pth"
            torch.save(model.state_dict(), model_path)

    # model_path = save_dir / "model.pth"
    swa_model_path = save_dir / "swa_model.pth"
    torch.save(model.state_dict(), model_path)
    torch.save(swa_model.state_dict(), swa_model_path)
    
    return results

def save_results(save_dir, results):
    result_path = save_dir / 'result.pkl'

    with open(result_path, 'wb') as f:
        pickle.dump(results, f)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()

    # data augmentation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # download dataset
    dataset_root = Path(args.data_path)
    dataset_path = load_dataset(data_path=dataset_root)

    def train_transform(image, mask, trimap):
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degree=(0, 180)),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        image_transform = transforms.Compose([
            transforms.Normalize(mean=[0.478, 0.446, 0.396],
                                 std=[0.226, 0.223, 0.225])
        ])

        transformed_sample = [image_transform(data_transform(image)), data_transform(mask), data_transform(trimap)]
        return transformed_sample

    def valid_transform(image, mask, trimap):
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        image_transform = transforms.Compose([
            transforms.Normalize(mean=[0.478, 0.446, 0.396],
                                 std=[0.226, 0.223, 0.225])
        ])

        transformed_sample = [image_transform(data_transform(image)), data_transform(mask), data_transform(trimap)]
        return transformed_sample


    train_data = OxfordPetDataset(root=dataset_path,
                                  mode='train',
                                  transform=train_transform)
    valid_data = OxfordPetDataset(root=dataset_path,
                                  mode='valid',
                                  transform=valid_transform)

    save_dir1 = Path("../saved_models/model_UNet_5")
    save_dir2 = Path("../saved_models/model_Res_UNet_4")

    UNet_results = train(args=args, 
                         train_data=train_data,
                         valid_data=valid_data,
                         net='UNet',
                         save_dir=save_dir1)
    Res_UNet_results = train(args=args, 
                             train_data=train_data,
                             valid_data=valid_data,
                             net='Res_UNet',
                             save_dir=save_dir2)
    
    save_results(save_dir1, UNet_results)
    save_results(save_dir2, Res_UNet_results)