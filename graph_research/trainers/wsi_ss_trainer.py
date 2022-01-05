import torch
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from pycm import *
from tqdm import tqdm
from torchvision import transforms, utils

from models import get_model
from trainers.base_trainer import BaseTrainer
from loaders import get_wsicompressed
from utils.misc import AverageMeter, adjust_lr_staircase, convert_state_dict
import trainers.wsi_seam_utils as visualization
from histomicstk.saliency.tissue_detection import get_tissue_mask


class WSISSTrainer(BaseTrainer):

    def __init__(self, cfg, writer, logger):
        super().__init__(cfg, writer, logger)

        # setup datasets
        print('Setting up data ...')
        self.data_transforms = {
            'train': transforms.Compose([transforms.ToTensor()]),
            'val': transforms.Compose([transforms.ToTensor()]),
        }

        loaders_dict = get_wsicompressed(self.cfg, self.data_transforms,
                                         use_json=True,
                                         use_multilab=True,
                                         train=True,
                                         mask_dir=self.cfg['data']['cam_path'])
                                         
        self.train_dset, self.train_loader = loaders_dict['train']
        self.val_dset, self.val_loader = loaders_dict['val']

        finetuned_params = list(self.model.module.features.parameters())
        new_params = [p for n, p in self.model.module.named_parameters()
                      if not n.startswith('features.')]

        param_groups = [{'params': finetuned_params, 'lr': self.cfg['training']['lr']},
                        {'params': new_params,       'lr': self.cfg['training']['fc_lr']}]

        self.optimizer = optim.Adam(param_groups)
        self.cls_n = self.cfg['arch']['n_classes']
        self.best_loss = 0.0
        
        masked    = self.cfg['training']['masked']
        self.w_c  = self.cfg['training']['w_c']
        self.w_h  = self.cfg['training']['w_h']
        self.w_er = self.cfg['training']['w_er']

        print(f'MASKED        ::: {masked}')
        print(f'WEIGHTS(LOSS) ::: L_C x {self.w_c} +  L_H x {self.w_h} + L_ER x {self.w_er}')

    def _train_epoch(self, epoch):

        losses_cls  = AverageMeter()
        losses_er   = AverageMeter()
        losses_cent = AverageMeter()
        losses_kl   = AverageMeter()
        train_accuracy = AverageMeter()

        adjust_lr_staircase(
            self.optimizer.param_groups,
            [self.cfg['training']['lr'], self.cfg['training']['fc_lr']],
            epoch + 1,
            [self.cfg['training']['epochs']//2, self.cfg['training']['epochs']],
            0.1
        )

        pbar = tqdm(self.train_loader, ncols=160, desc=' ')

        for i, data in enumerate(pbar):
            inputs = data[0]
            label  = data[1]

            img   = inputs.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            label = label.unsqueeze(2).unsqueeze(3)

            f_d = self.model.module.features(img)
            #size (64,64)
            cam, cam_rv, valid_pixels = self.model.module.pooling(
                f_d, img, size=(64, 64), masked=self.cfg['training']['masked'],
                train=True, scratch=True)

            label_pred_cam_or = F.adaptive_avg_pool2d(cam, (1, 1))
            label_pred_cam_rv = F.adaptive_avg_pool2d(cam_rv, (1, 1))

            cam = visualization.max_norm(cam)  * label
            cam_rv = visualization.max_norm(cam_rv)  * label

            loss_cls1 = F.multilabel_soft_margin_loss(label_pred_cam_or, label)
            loss_cls2 = F.multilabel_soft_margin_loss(label_pred_cam_rv, label)

            # Equivarinat loss
            loss_er = torch.mean(
                torch.abs(cam[:, 1:, :, :] - cam_rv[:, 1:, :, :])) * self.w_er

            # conditional entropy: helps +0.10% improvement
            cond_entropy = - \
                ((valid_pixels * (cam_rv * torch.log(cam + 1e-10))).sum(1))
            cond_entropy = cond_entropy.sum(dim=(1, 2))
            cond_entropy /= valid_pixels.squeeze(1).sum(dim=(1, 2))
            cond_entropy = cond_entropy.mean(0) * self.w_h

            # Classification loss
            loss_cls = ((loss_cls1 + loss_cls2) / 2) * self.w_c

            # Total Loss
            loss = loss_cls + loss_er + cond_entropy
            ##
            loss.backward()
            self.optimizer.step()

            losses_cls.append(loss_cls.item())
            losses_er.append(loss_er.item())
            losses_cent.append(cond_entropy.item())

            pbar.set_description('--(train) | CLS {:.6f} | ER: {:.6f} | H: {:.6f} '.format(
               losses_cls.avg(), losses_er.avg(), losses_cent.avg())
            )

        self.train_loss = (losses_cls.avg() + losses_er.avg() +
                           losses_cent.avg())/(1.0 + self.w_h + self.w_er)
        step = epoch + 1
        self.writer.add_scalar('training/loss_cls', losses_cls.avg(), step)
        self.writer.add_scalar('training/loss_er',  losses_er.avg(),  step)
        self.writer.add_scalar('training/loss_h',   losses_cent.avg(), step)

        print()

    def _valid_epoch(self, epoch):
        print(f'Loss -- {self.train_loss}')
        return {'loss': self.train_loss}


    def _on_train_start(self):
        pass

    def _on_train_end(self):
        pass

    def _on_valid_start(self):
        pass

    def _on_valid_end(self):
        pass
