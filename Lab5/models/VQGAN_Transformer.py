import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        z_q, indices, q_loss = self.vqgan.encode(x)
        indices = indices.view(z_q.shape[0], -1)
        # print(indices.shape)
        return z_q, indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            print('linear')
            return lambda r: 1 - r
        elif mode == "cosine":
            print('cosine')
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            print('square')
            return lambda r: 1 - r ** 2
        else:
            raise NotImplementedError
        
    def generate_mask(self, size):
        mask = np.zeros(size, dtype=np.int32)
        x, y = size[0] // 2, size[1] // 2  # Start from the center point
        
        for _ in range(size[0] * size[1] // 2): 
            mask[x, y] = 1
            # Randomly move in one of the four directions
            direction = np.random.choice(['up', 'down', 'left', 'right'])
            if direction == 'left' and x > 0:
                x -= 1
            elif direction == 'right' and x < size[0] - 1:
                x += 1
            elif direction == 'down' and y > 0:
                y -= 1
            elif direction == 'up' and y < size[1] - 1:
                y += 1
        return mask

##TODO2 step1-3:            
    def forward(self, x):
        _, z_indices = self.encode_to_z(x)
        
        # r = math.floor(self.gamma(np.random.uniform()) * z_indices.shape[1])
        # sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
        # mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
        # mask.scatter_(dim=1, index=sample, value=True)
        mask_size = (16, 16)
        mask = self.generate_mask(mask_size)
        mask = torch.tensor(mask, dtype=torch.bool).view(-1).unsqueeze(0).to(z_indices.device)

        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)
        a_indices = (~mask) * z_indices + mask * masked_indices

        logits = self.transformer(a_indices)
        # print(logits.shape, z_indices.shape)
        return logits, z_indices
    

##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask_b, mask_bc, total_iter, current_iter):

        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)
        a_indices = mask_bc * masked_indices + (~mask_bc) * z_indices

        # Apply the transformer model to generate logits
        logits = self.transformer(a_indices)

        # Apply softmax to convert logits into a probability distribution accross the last dimension
        probs = torch.softmax(logits, dim=-1)

        # Find the maximum probability for each token value
        max_probs, z_indices_predict = torch.max(probs, dim=-1)

        # Calculate temperature annealnig for confidence
        ratio = (current_iter) / (total_iter)
        temperature = self.choice_temperature * (1 - ratio)

        # Generate GumBel noise
        gumbel_noise = torch.rand_like(max_probs).log().neg().log().neg()

        # predicted probabilities add temperature annealing gumbel noise as confidence
        confidence = max_probs + temperature * gumbel_noise

        # Apply the mask to the confidence values
        confidence_masked = confidence.masked_fill((~mask_bc), float('inf'))
        # Sort the confidence for the rank
        _, sorted_indices = torch.sort(confidence_masked, dim=-1)

        # scheduling strategy
        mask_num = int(torch.sum(mask_b))
        num_remain_tokens = int(self.gamma(ratio)*mask_num)

        # Create a mask indicating which tokens to keep and which to mask
        mask = torch.zeros_like(mask_b)
        mask.scatter_(1, sorted_indices[:, :num_remain_tokens], 1)

        diff = (mask != mask_bc)

        # At the end of the decoding process, add back the original token that were not masked to the predicted tokens
        original_tokens = z_indices.clone()
        # print(original_tokens.shape, sorted_indices.shape)
        final_tokens = torch.where(mask, z_indices_predict, original_tokens)
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        
        # z_indices = (~diff) * z_indices + diff * z_indices_predict

        return final_tokens, mask, z_indices
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
