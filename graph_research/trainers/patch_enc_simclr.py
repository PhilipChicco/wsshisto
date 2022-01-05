import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy, numpy as np

from tqdm import tqdm
from torchvision import transforms

from trainers.base_trainer import BaseTrainer
from loaders import get_wsifolders
from utils.misc import AverageMeter
from utils.data_aug import GaussianBlur


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size  = batch_size
        self.temperature = temperature
        self.device = device

        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def recall_rpr(self):
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, representations):
        representations = F.normalize(representations,dim=1)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class PatchEncoder(BaseTrainer):

    def __init__(self, cfg, writer, logger):
        super().__init__(cfg, writer, logger)

        # setup datasets
        print('Setting up data ...')
        size = 256
        color_jitter = transforms.ColorJitter(0.25, 0.25, 0.25, 0.25)
        data_trans   = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),]
                        )
        self.data_transforms = {
            'train': {
                'orig' : transforms.Compose([transforms.Resize(size),transforms.ToTensor()]),
                'aug'  : data_trans,
            },
            'val': {
                'orig' : transforms.Compose([transforms.Resize(size),transforms.ToTensor()]),
                'aug'  : data_trans,
            },
        }

        loaders_dict = get_wsifolders(self.cfg, self.data_transforms, patch=True, use_sampler=False)
        self.train_dset, self.train_loader = loaders_dict['train']
        #self.val_dset  , self.val_loader   = loaders_dict['val']

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['training']['lr'])
        self.ntc_loss  = NTXentLoss(self.device, self.cfg['training']['train_batch_size'],
                                   temperature=0.5, use_cosine_similarity=True)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                              T_max=len(self.train_loader),
                                                              eta_min=0,
                                                              last_epoch=-1)
        self.current_loss = 0.
        print()
        print(self.model.module.pooling)
        print()

    def _train_epoch(self, epoch):
        logits_losses  = AverageMeter()

        pbar = tqdm(self.train_loader, ncols=160, desc=' ')
        for i, data in enumerate(pbar):
            inputs = torch.cat([data[0], data[1]], 0).to(self.device)
            self.optimizer.zero_grad()

            features = self.model(inputs)
            loss = self.ntc_loss(features)
            
            logits_losses.append(loss.item())

            loss.backward()
            self.optimizer.step()

            pbar.set_description(
                '--- (train) | Loss: {:.6f}  :'.format(
                    logits_losses.avg(),
                )
            )

        step = epoch + 1
        self.writer.add_scalar('training/loss', logits_losses.avg(), step)
        print()
        self.current_loss = logits_losses.avg()

    def _valid_epoch(self, epoch):
        return {'loss': self.current_loss}

    def _on_train_start(self):
        pass

    def _on_train_end(self):
        self.scheduler.step()

    def _on_valid_start(self):
        pass

    def _on_valid_end(self):
        pass


