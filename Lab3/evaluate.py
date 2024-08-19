import torch
from utils import dice_score

def evaluate(net, data, device):
    # implement the evaluation function here
    valid_acc = 0
    net.eval()
    with torch.inference_mode():
        for (images, masks, trimaps) in data:
            images, masks = images.to(device), masks.to(device)
    
            outputs = net(images)

            output_logit = torch.sigmoid(outputs)
            # output_pred = torch.round(outputs)
            # valid_acc += (output_pred == masks).float().mean().item()
                
            valid_acc += dice_score(torch.round(output_logit), masks)

    valid_acc = valid_acc / len(data)
    return valid_acc