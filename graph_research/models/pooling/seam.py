import torch, math, numpy as np
import torch.nn as nn
import torch.nn.functional as F


class WSISS(nn.Module):
    def __init__(self, in_channels, classes=None, embed=64, channels=64):
        super().__init__()
        in_channels = 64
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip([6, 12, 18, 24], [6, 12, 18, 24]):
            self.conv2d_list.append(
                nn.Conv2d(
                    in_channels, # 64 
                    classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                )
            )

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)
        
        # theta PCM
        self.pcm_head = torch.nn.Conv2d(in_channels, in_channels,  1, bias=False)        
        torch.nn.init.xavier_uniform_(self.pcm_head.weight)
        

    def forward(self, d, img, size=(128,128), out_size=(256,256),
        masked=False,train=False, scratch=False):

        N, C, H, W = img.size()
        if scratch:
            Nd, Cd, Hd, Wd = d.size()
            cam = self.conv2d_list[0](F.relu(d))
            for i in range(len(self.conv2d_list) - 1):
                cam += self.conv2d_list[i + 1](F.relu(d))
        else:
            cam = self.pcm_head(F.relu(d['conv1']))

        n, c, h, w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1) + 1e-5
            cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
            cam_d_norm[:, 0, :, :] = 1 - torch.max(cam_d_norm[:, 1:, :, :], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:, 1:, :, :], dim=1, keepdim=True)[0]
            cam_d_norm[:, 1:, :, :][cam_d_norm[:, 1:, :, :] < cam_max] = 0
        
        if masked:
            max_ones = []
            thr_list = [0.2,0.3,0.4]
            for i in thr_list:
                max_ones.append(self.max_onehot(cam_d_norm.clone(),x_max=i).unsqueeze(1))
            max_onehot = torch.mean(torch.cat(max_ones,1).float(),dim=1)
            img = img * max_onehot.long().unsqueeze(1)
        else:
            max_onehot = torch.ones((n,h,w)).long().cuda()

        cam_rv = F.interpolate(self.PCM(cam_d_norm, img.detach(),size,out_size), (H, W), mode='bilinear', align_corners=True)
        cam    = F.interpolate(cam, (H, W), mode='bilinear', align_corners=True)

        if train:
            return cam, cam_rv, max_onehot.unsqueeze(1)
        else:
            return cam, cam_rv

    def max_onehot(self, x,x_max=0.2):
        x[:, 1:, :, :][x[:, 1:, :, :] < x_max]  = 0.0
        x[:, 1:, :, :][x[:, 1:, :, :] >= x_max] = 0.9
        return x.argmax(1).long()

    def PCM(self, cam, f, size=(128,128), out_size=(256,256)):
        f = self.pcm_head(f)
        cam = F.interpolate(cam, size=size, mode='bilinear', align_corners=True)
        f   = F.interpolate(f,   size=size, mode='bilinear', align_corners=True)
        n, c, h, w = cam.size()
        cam = cam.view(n, -1, h * w)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True)+ 1e-5)
        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff / (torch.sum(aff, dim=1, keepdim=True)+ 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)
        cam_rv = F.interpolate(cam_rv, size=out_size, mode='bilinear', align_corners=True)
        return cam_rv


    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, 1)

    @staticmethod
    def predictions(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(1)

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor):
        return F.multilabel_soft_margin_loss(logits, labels)