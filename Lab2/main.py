import torch
from torch import nn
from torchinfo import summary
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import random
import numpy as np
from VGG19 import VGG19 as VGG
from ResNet50 import ResNet50 as ResNet
from dataloader import BufferflyMothLoader as BM
from tqdm.auto import tqdm

def evaluate(model: torch.nn.Module,
             dataloader,
             loss_fn: torch.nn.Module,
             device):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    eval_loss, eval_acc = 0, 0

    num_data = 0

    # turn on inference mode
    with torch.inference_mode():
        for batch, (x, y) in enumerate(dataloader):
            # send data to the target device
            x, y = x.to(device), y.to(device)

            # forward pass
            eval_pred_logits = model(x)

            # calculate the loss
            loss = loss_fn(eval_pred_logits, y)
            eval_loss += loss.item()

            # calculate the accuracy
            eval_pred_class = torch.argmax(torch.softmax(eval_pred_logits, dim=1), dim=1)
            eval_acc += (eval_pred_class == y).sum().item()

            num_data += len(eval_pred_class)

    # Adjust metrics to get average loss and accuracy 
    eval_loss = eval_loss / len(dataloader)
    eval_acc = eval_acc / num_data
    return eval_loss, eval_acc
    

def test(model: torch.nn.Module,
         dataloader,
         device):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_acc = 0

    num_data = 0

    # turn on inference mode
    with torch.inference_mode():
        for batch, (x, y) in enumerate(dataloader):
            # send data to the target device
            x, y = x.to(device), y.to(device)

            # forward pass
            test_pred_logits = model(x)

            # calculate the accuracy
            test_pred_class = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
            test_acc += (test_pred_class == y).sum().item()

            num_data += len(test_pred_class)

    # Adjust metrics to get average accuracy 
    test_acc = test_acc / num_data
    return test_acc
    
    

def train(model: torch.nn.Module,
          dataloader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device):
    # Put the model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    num_data = 0

    for batch, (x, y) in enumerate(dataloader):
        # Send data to the target device
        x, y = x.to(device), y.to(device)

        # forward pass
        y_pred = model(x)

        # calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # calculate accuracy metric
        # print(y_pred)
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()

        num_data += len(y_pred_class)
        if batch % 10 ==0:
            print(batch)
        

    # Adjust metrics to get average loss and accuracy
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / num_data
    return train_loss, train_acc
        


if __name__ == "__main__":
    # print("Good Luck :)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(36)
    torch.cuda.manual_seed(36)

    train_data = BM(root='./dataset/',
                    mode='train')
    valid_data = BM(root='./dataset/',
                    mode='valid')
    test_data = BM(root='./dataset/',
                    mode='test')
    
    BATCH_SIZE = 256
    NUM_WORKERS = 0
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True,
                                  shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_data,
                                  batch_size=BATCH_SIZE,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True,
                                  shuffle=False)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS,
                                 pin_memory=True,
                                 shuffle=False)
    
    # print(len(train_dataloader), len(valid_dataloader), len(test_dataloader))


    model_vgg = VGG(input_shape=3,
                    output_shape=100).to(device)
    # summary(model_vgg, input_size=[1, 3, 224, 224])

    model_resnet = ResNet(input_shape=3,
                          output_shape=100).to(device)
    # summary(model_resnet, input_size=[1, 3, 224, 224])

    results = {'train_loss': [],
               'train_acc': [],
               'valid_loss': [],
               'valid_acc': []}
    epochs = 5

    # Set loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer_vgg = torch.optim.Adam(params=model_vgg.parameters(),
                                 lr=0.001)
    
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train(model=model_vgg,
                                      dataloader=train_dataloader,
                                      loss_fn=loss_fn,
                                      optimizer=optimizer_vgg,
                                      device=device)
        eval_loss, eval_acc = evaluate(model=model_vgg,
                                       dataloader=valid_dataloader,
                                       loss_fn=loss_fn,
                                       device=device)
        
        # print out loss and accuracy
        print(f"Epoch: {epoch} | train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | eval loss: {eval_loss:.4f} | eval acc: {eval_acc:.4f}")

        results['train_acc'].append(train_acc)
        results['train_loss'].append(train_loss)
        results['valid_acc'].append(eval_acc)
        results['valid_loss'].append(eval_loss)
    