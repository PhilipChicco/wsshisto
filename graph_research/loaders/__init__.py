import os, torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from . import *
from .wsi_datasets import WSIFolders, SlideCompressed

import numpy as np


loaders = {
    # wsifolders
    'wsifolders'    : WSIFolders,
    'wsicompressed' : SlideCompressed,
    
}


# wsi funcs
def get_wsifolders(cfg, data_transforms, patch=False, use_sampler=True):
    state_shuff = True if patch == True else False
    print(f'Shuffle is set to {state_shuff}')
    print(f'Weighted-Sampler is set to {use_sampler}')
    class_map = cfg['data']['classmap']
    data_path = cfg['data']['data_path']
    data_loader = loaders[cfg['data']['dataset']]

    t_dset = data_loader(
        root=data_path,
        split=cfg['data']['train_split'],
        transform=data_transforms['train'],
        class_map=class_map,
        nslides=cfg['data']['nslides']
    )

    if use_sampler:
        count_dict = get_class_distribution(cfg, t_dset)
        target_list = torch.tensor(t_dset.slideLBL)
        target_list = target_list[torch.randperm(len(target_list))]
        class_count = [i for i in count_dict.values()]
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        class_weights_all = class_weights[target_list]

        weighted_sampler = WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
        )

        t_loader = DataLoader(t_dset,
                              batch_size=cfg['training']['train_batch_size'],
                              num_workers=cfg['training']['n_workers'],
                              shuffle=False,  # should be false for patch based training with sampler
                              pin_memory=False, sampler=weighted_sampler,
                              drop_last=True
                              )
    else:
        t_loader = DataLoader(t_dset,
                              batch_size=cfg['training']['train_batch_size'],
                              num_workers=cfg['training']['n_workers'],
                              shuffle=state_shuff,  # should be false for patch based training
                              pin_memory=False,
                              drop_last=True
                              )

    v_dset = data_loader(
        root=data_path,
        split=cfg['data']['val_split'],
        transform=data_transforms['val'],
        class_map=class_map,
        nslides=cfg['data']['nslides']
    )
    v_loader = DataLoader(v_dset,
                          batch_size=cfg['training']['val_batch_size'],
                          num_workers=cfg['training']['n_workers'],
                          shuffle=False, pin_memory=False,
                          drop_last=True
                          )

    return {'train': (t_dset, t_loader), 'val': (v_dset, v_loader)}

def get_class_distribution(cfg, dataset_obj):
    count_dict = {x: 0 for idx, x in enumerate(cfg['data']['classmap'].split(","))}
    class_map = {idx: x for idx, x in enumerate(cfg['data']['classmap'].split(","))}

    for element in dataset_obj.slideLBL:
        y_lbl = class_map[element]
        count_dict[y_lbl] += 1

    return count_dict

# wsi compressed
def get_wsicompressed(cfg, data_transforms, train=True, use_json=False, use_multilab=False, mask_dir=None):
    class_map = cfg['data']['classmap']
    data_path = cfg['data']['data_path']
    data_loader = loaders[cfg['data']['dataset']]

    t_dset = data_loader(
        root=data_path,
        split=cfg['data']['train_split'],
        transform=data_transforms['train'],
        class_map=class_map,
        nslides=cfg['data']['nslides'],
        train=train,
        use_json=use_json,
        use_multilab=use_multilab,
        mask_dir=mask_dir
    )

    weights = get_class_distribution_slides(cfg, t_dset)
    weights = torch.DoubleTensor(weights)
    print('SAMPLER WEIGHTS :::: ',len(weights), np.unique(weights.data.cpu().numpy()))
    
    weighted_sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights)
    )

    t_loader = DataLoader(t_dset,
                          batch_size=cfg['training']['train_batch_size'],
                          num_workers=cfg['training']['n_workers'],
                          sampler=weighted_sampler,
                          shuffle=False,  # should be false for patch based training with sampler
                          pin_memory=False,
                          )

    v_dset = data_loader(
        root=data_path,
        split=cfg['data']['val_split'],
        transform=data_transforms['val'],
        class_map=class_map,
        nslides=cfg['data']['nslides'],
        train=False,
        use_json=use_json,
        use_multilab=use_multilab,
        mask_dir=mask_dir
    )
    v_loader = DataLoader(v_dset,
                          batch_size=cfg['training']['val_batch_size'],
                          num_workers=cfg['training']['n_workers'],
                          shuffle=False, pin_memory=False
                          )

    return {'train': (t_dset, t_loader), 'val': (v_dset, v_loader)}

def get_wsicompressed_test(cfg, data_transforms,use_json=True,mask_dir=None):
    class_map = cfg['data']['classmap']
    data_path = cfg['data']['data_path']
    data_loader = loaders[cfg['data']['dataset']]

    t_dset = data_loader(
        root=data_path,
        split=cfg['data']['test_split'],
        transform=data_transforms['test'],
        class_map=class_map,
        nslides=cfg['data']['nslides'],
        train=False,
        use_json=use_json,
        mask_dir=mask_dir
    )

    t_loader = DataLoader(t_dset,
                          batch_size=cfg['training']['test_batch_size'],
                          num_workers=cfg['training']['n_workers'],
                          shuffle=False,  # should be false for patch based training with sampler
                          pin_memory=False,
                          )

    return {'test': (t_dset, t_loader)}

def get_class_distribution_slides(cfg, dset):
    count = [0] * cfg['arch']['n_classes']
    for item in dset.targets:
        count[item] += 1
        #print('COUNT CLASSES : ', count)
    weight_per_class = [0.] * cfg['arch']['n_classes']
    N = float(sum(count))
    for i in range(cfg['arch']['n_classes']):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(dset.targets)
    for idx, val in enumerate(dset.targets):
        weight[idx] = weight_per_class[val]
    return weight


datamethods = {
    'wsifolders'   : get_wsifolders,
    'wsicompressed': get_wsicompressed,
}

    

    



#

