from .patch_enc_simclr import PatchEncoder
from .wsi_seg_trainer import WSISegTrainer
from .wsi_ss_trainer import WSISSTrainer


trainers_dict = {
    
    'patchencoder' : PatchEncoder,
    'wsiseg'       : WSISegTrainer,
    'wsiss'        : WSISSTrainer,
}

def get_trainer(name):
    names = list(trainers_dict.keys())
    if name not in names:
        raise ValueError('Invalid choice for trainers - choices: {}'.format(' | '.join(names)))
    return trainers_dict[name]