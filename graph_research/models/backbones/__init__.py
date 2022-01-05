

from .unet import unet_d5 as unet
from .resnet import resnet18 as resnet18
import torch.nn as nn

backbones = {
    
    # resnets
    'resnet18' : resnet18,

    # slides
    'slide_unet'   : unet,
    'identity'     : nn.Identity, # input dim should be None
    
    
}

def load_backbone(backbone, pretrained=True, input_dim=None, output_dim=None):
    net_names = list(backbones.keys())
    if backbone not in net_names:
        raise ValueError('Invalid choice for backbone - choices: {}'.format(' | '.join(net_names)))
    
    if not backbone.startswith('resnet'): 
        if 'slide_unet' in backbone:
            return backbones[backbone](in_channels=input_dim)

        if input_dim is None:
            return backbones[backbone]()
        else:
            return backbones[backbone](input_dim,output_dim)
    else:
        print('Loading pretrained backbone .............')
        return backbones[backbone](pretrained)
   






