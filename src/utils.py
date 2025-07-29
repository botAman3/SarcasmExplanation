import torch
import torch.nn as nn
from torchvision import models

def freeze_params(model: nn.Module):
    """
    Freezes the parameters of a model or its sub-modules.
    """
    for param in model.parameters():
        param.requires_grad = False

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shifts input ids one token to the right.
    """
    prev_output_tokens = input_ids.clone()
    assert pad_token_id is not None, "pad_token_id has to be defined."
    
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens
    
    return prev_output_tokens

def load_image_encoder():
    """
    Loads the feature extractor from a pre-trained VGG19 model.
    """
    vgg19model = models.vgg19(pretrained=True)
    image_encoder = list(vgg19model.children())[0]
    return image_encoder