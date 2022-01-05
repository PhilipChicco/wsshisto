import torch
import torch.nn as nn


def conv_module(in_channels,channels,norm_layer):
  conv1 = nn.Conv2d(in_channels=in_channels,out_channels=channels,kernel_size=3,padding=1)
  bn1 = norm_layer(channels)
  relu1 = nn.ReLU(inplace=True)
  conv2 = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1)
  bn2 = norm_layer(channels)
  relu2 = nn.ReLU(inplace=True)
  return nn.Sequential(conv1,bn1,relu1,conv2,bn2,relu2)


def convt_module(in_channels,channels,norm_layer):
  conv_tr = nn.ConvTranspose2d(in_channels=in_channels,out_channels=channels,kernel_size=2,stride=2)
  bn = norm_layer(channels)
  relu = nn.ReLU(inplace=True)
  return nn.Sequential(conv_tr,bn,relu)


def pool_module(pool_type,channels):
  if pool_type == "max":
    return nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
  elif pool_type == "avg":
    return nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
  else:
    return nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=2,padding=1)


class JoinOp(object):
  def __init__(self,join_type):
    self.join_type = join_type

  def __call__(self,x1,x2):
    if self.join_type == "concatenate":
      x = torch.cat((x1,x2),dim=1)
    elif self.join_type == "add":
      x = torch.add(x1,x2)
    elif self.join_type is None:
      x = x1
    else:
      self._attr_error()
    return x

  def get_in_channels(self,channels):
    if self.join_type == "concatenate":
      x = 2*channels
    elif self.join_type == "add":
      x = channels
    elif self.join_type is None:
      x = channels
    else:
      self._attr_error()
    return x

  def _attr_error(self):
    raise AttributeError(self.join_type +
      " is not a supported op to join encoder and decoder features")


class Encoder(nn.Module):
  def __init__(self,cfg,in_channels,norm_layer,pool_type):
    super(Encoder, self).__init__()
    self.depth = len(cfg)
    convs = [0]*self.depth
    pools = [0]*(self.depth-1)
    channels = cfg[0]
    self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=channels,kernel_size=5,padding=2)
    self.bn1 = norm_layer(channels)
    self.relu1 = nn.ReLU(inplace=True)
    in_channels = cfg[0]
    for d in range(self.depth):
      channels = cfg[d]
      convs[d] = conv_module(in_channels=in_channels,channels=channels,norm_layer=norm_layer)
      if d<(self.depth-1):
        pools[d] = pool_module(pool_type=pool_type,channels=channels)
      in_channels = cfg[d]
    self.convs = nn.ModuleList(convs)
    self.pools = nn.ModuleList(pools)

  def forward(self,x):
    features = {}
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)
    for d in range(self.depth):
      x = self.convs[d](x)
      features["e"+str(d)] = x
      if d<(self.depth-1):
        x = self.pools[d](x)
    return features


class Decoder(nn.Module):
  def __init__(self,cfg,norm_layer,join_op):
    super(Decoder, self).__init__()
    self.depth = len(cfg)
    convts = [0]*(self.depth-1)
    convs = [0]*(self.depth-1)
    self.join_op = join_op
    for d in range(self.depth-1):
      in_channels = cfg[d+1]
      channels = cfg[d]
      convts[d] = convt_module(in_channels=in_channels,channels=channels,norm_layer=norm_layer)
      in_channels = join_op.get_in_channels(channels)
      convs[d] = conv_module(in_channels=in_channels,channels=channels,norm_layer=norm_layer)
    self.convts = nn.ModuleList(convts)
    self.convs = nn.ModuleList(convs)

  def forward(self,features):
    x = features["e"+str(self.depth-1)]
    for d in range(self.depth-2,-1,-1):
      x2 = features["e"+str(d)]
      x = self.convts[d](x)
      x = self.join_op(x,x2)
      x = self.convs[d](x)
      features["d"+str(d)] = x
    return features


class Classifier(nn.Module):
  def __init__(self,in_channels,num_classes):
    super(Classifier, self).__init__()
    self.conv = nn.Conv2d(in_channels=in_channels,out_channels=num_classes,kernel_size=1,bias=False)

  def forward(self, features):
    x = self.conv(features["d0"])
    features["out"] = x
    return features

class ViewFlatten(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,x):
    return x.view(x.shape[0],-1)

class Unet(nn.Module):
  def __init__(self,cfg,in_channels,num_classes,norm_layer=None,pool_type=None,join_type=None,return_features=False):
    super(Unet, self).__init__()
    self.return_features = return_features
    self.depth = len(cfg)
    if norm_layer is None:
      self.norm_layer = nn.Identity
    else:
      self.norm_layer = norm_layer

    self.join_op = JoinOp(join_type)
    self.encoder = Encoder(cfg=cfg,in_channels=in_channels,norm_layer=self.norm_layer,pool_type=pool_type)
    self.decoder = Decoder(cfg=cfg,norm_layer=self.norm_layer,join_op=self.join_op)
    self.classifier = Classifier(in_channels=cfg[0],num_classes=num_classes)

  def forward(self,x):
    efeatures = self.encoder(x)
    dfeatures = self.decoder(efeatures)
    out       = self.classifier(dfeatures)
    return out["out"]


def _unet(cfg, in_channels, num_classes, **kwargs):
  model = Unet(cfg=cfg, in_channels=in_channels, num_classes=num_classes, **kwargs)
  return model


def unet_d4(in_channels, num_classes, return_features=False):
  cfg = [64,128,256,512]
  model = _unet(cfg=cfg, in_channels=in_channels, num_classes=num_classes,
    pool_type="max",join_type="concatenate",return_features=return_features)
  return model

def unet_d5(in_channels=64, num_classes=2, return_features=False):
  cfg = [64,96,128,256,512]
  norm_layer = nn.BatchNorm2d
  model = _unet(cfg=cfg, in_channels=in_channels, num_classes=num_classes,
    pool_type="max",join_type="concatenate",return_features=return_features,norm_layer=norm_layer)
  return model

if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
    from utils.misc import print_network

    inp = torch.randn((1,64,256,256))

    net = unet_d5(64,1)
    print_network(net)

    out, enc = net(inp)

    print(f'in {inp.shape} out {out.shape} {enc.shape}')