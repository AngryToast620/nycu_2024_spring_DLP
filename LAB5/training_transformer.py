import warnings

warnings.filterwarnings("ignore", message="Failed to load image Python extension", category=UserWarning)

import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from pathlib import Path


#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim, self.scheduler = self.configure_optimizers()
        self.prepare_training()
        
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_dataset, step):
        print("Training Transformer")
        with tqdm(range(len(train_dataset))) as pbar:
            total_loss = 0
            # self.scheduler.step()
            for i, imgs in zip(pbar, train_dataset):
                imgs = imgs.to(self.args.device)
                # print(imgs.shape)
                logits, target = self.model(imgs)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                total_loss += loss
                loss.backward()
                # if step % self.args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()
                # self.scheduler.step()   
                # step += 1
                pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4), LR=np.round(self.optim.param_groups[0]['lr'], 6))
                pbar.update(0)
            self.scheduler.step(total_loss)
            print(f"----------Loss: {total_loss/len(train_dataset):.3f} | lr: {self.optim.param_groups[0]['lr']}-----------")
            

    def eval_one_epoch(self, val_dataset):
        print("Evaluating Transformer")
        with torch.inference_mode():
            with tqdm(range(len(val_dataset))) as pbar:
                total_loss = 0
                for i, imgs in zip(pbar, val_dataset):
                    imgs = imgs.to(self.args.device)
                    logits, target = self.model(imgs)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                    total_loss += loss
                    pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                    pbar.update(0)

                print(f"----------{total_loss/len(val_dataset):.3f}-----------")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.transformer.parameters(), lr=0.0001, betas=(0.9, 0.96), weight_decay=4.5e-2)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=self.args.epochs*375, three_phase=True)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, min_lr=1e-6, patience=2)
        # scheduler = None
        return optimizer, scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./transformer_checkpoints/square/transformer_current.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum_grad', type=int, default=25, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
    parser.add_argument('--save_per_epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start_from_epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt_interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= Path(args.train_d_path), partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= Path(args.val_d_path), partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5: 
    step = args.start_from_epoch * len(train_dataset)   
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        print(f'Epoch {epoch}:')
        train_transformer.train_one_epoch(train_loader, step)

        train_transformer.eval_one_epoch(val_loader)

        if epoch % args.save_per_epoch == 0:
            torch.save(train_transformer.model.transformer.state_dict(), os.path.join("transformer_checkpoints", 'cos1', f"transformer_epoch_{epoch}.pt"))
        torch.save(train_transformer.model.transformer.state_dict(), os.path.join("transformer_checkpoints", 'cos1', "transformer_current.pt"))