# backbones
import torch.nn as nn
from collections import OrderedDict

from models.backbones import load_backbone
from models.pooling import load_pooling


def get_model(cfg):
    
    backbone  = cfg['arch']['backbone']
    pooling   = cfg['arch']['pooling']
    n_classes = cfg['arch']['n_classes']
    embed     = cfg['arch']['embedding']

    if backbone in ['slide_unet','identity']:
        input_dim    = cfg['arch']['input_dim']
        backbone     = load_backbone(backbone,input_dim=input_dim, output_dim=embed)
        out_channels = embed
    elif backbone in ['resnet_slide']:
        input_dim = cfg['arch']['input_dim']
        backbone = load_backbone(backbone, input_dim=input_dim, output_dim=embed)
        out_channels = embed
        backbone.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        backbone     = load_backbone(backbone)
        out_channels = backbone.inplanes


    pooling = load_pooling(pooling, out_channels, n_classes, embed)

    # final model
    model = nn.Sequential(OrderedDict([
            ('features', backbone),
            ('pooling' , pooling)
        ]))

    return model
