import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy, numpy as np

from tqdm import tqdm
from torchvision import transforms

from trainers.base_trainer import BaseTrainer
from loaders import get_wsicompressed
from utils.misc import AverageMeter, adjust_lr_staircase


class WSISegTrainer(BaseTrainer):

    def __init__(self, cfg, writer, logger):
        super().__init__(cfg, writer, logger)

        # setup datasets
        print('Setting up data ...')
        self.data_transforms = {
            'train': transforms.Compose([transforms.ToTensor()]),
            'val'  : transforms.Compose([transforms.ToTensor()]),
        }

        loaders_dict = get_wsicompressed(self.cfg,
                                         self.data_transforms,
                                         use_json=True,
                                         train=True, # True
                                         mask_dir=self.cfg['data']['cam_path'])
        self.train_dset, self.train_loader = loaders_dict['train']
        self.val_dset  , self.val_loader   = loaders_dict['val']

        finetuned_params = list(self.model.module.features.parameters())
        new_params       = [p for n, p in self.model.module.named_parameters()
                      if not n.startswith('features.')]

        param_groups = [{'params': finetuned_params, 'lr': self.cfg['training']['lr']},
                        {'params': new_params, 'lr': self.cfg['training']['fc_lr']}]

        self.optimizer  = optim.Adam(param_groups)

        print()
        

    def _train_epoch(self, epoch):
        seg_losses     = AverageMeter()

        adjust_lr_staircase(
            self.optimizer.param_groups,
            [self.cfg['training']['lr'], self.cfg['training']['fc_lr']],
            epoch + 1,
            [self.cfg['training']['epochs']//2],
            0.1
        )

        pbar = tqdm(self.train_loader, ncols=160, desc=' ')

        for i, data in enumerate(pbar):
            inputs = data[0]
            mask   = data[2]

            inputs = inputs.to(self.device)
            mask   = mask.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(inputs)

            loss   = self.model.module.pooling.loss(logits, mask)
            seg_losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            pbar.set_description('--- (train) | Loss[SEG]: {:.6f}  :'.format(
                seg_losses.avg())
            )

        step = epoch + 1
        self.writer.add_scalar('training/loss', seg_losses.avg(), step)
        print()

    def _valid_epoch(self, epoch):
        val_loss     = AverageMeter()

        with torch.no_grad():
            final_itr = tqdm(self.val_loader, ncols=80, desc=' ')

            for i, data in enumerate(final_itr):
                inputs = data[0]
                labels = data[2]

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(inputs)
                loss    = self.model.module.pooling.loss(logits, labels)
                val_loss.append(loss.item())

                final_itr.set_description('--- (val) | Loss : {:.6f}  :'.format(
                    val_loss.avg())
                )

        err_loss = val_loss.avg()

        # log
        self.writer.add_scalar('validation/loss' , err_loss, epoch)
        print()
        print('---> | loss : {:.4f} '.format(err_loss))
        return {'loss': err_loss}

    def _on_train_start(self):
        pass

    def _on_train_end(self):
        pass

    def _on_valid_start(self):
        pass

    def _on_valid_end(self):
        pass


