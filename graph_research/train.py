## Refer to MetricMWSI and SemiSupervised Repos

import warnings
warnings.filterwarnings("ignore")

import argparse
import yaml
import os,sys , torch, random, numpy as np

# trainers
from trainers import get_trainer
from utils.misc import get_logger

from tensorboardX import SummaryWriter

def main(cfg):
    print(cfg)
    print()

    # setup logdir, writer and logger
    logdir = os.path.join(cfg['root'], cfg['logdir'])
    os.makedirs(logdir, exist_ok=True)
    print(f'LOGDIR {logdir}')

    writer = SummaryWriter(log_dir=logdir)

    trainer_name = cfg['trainer']

    with open(os.path.join(logdir,trainer_name+'.yml'), 'w') as fp:
        yaml.dump(cfg, fp)

    logger  = get_logger(logdir,trainer_name)

    Trainer = get_trainer(trainer_name)(cfg, writer, logger)
    print()

    # start training
    Trainer.train()



if __name__ == '__main__':
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # get configs
    parser = argparse.ArgumentParser(description="Train a Network")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./graph_research/configs/wsi/wsi_ss_cm.yml",
        help="Configuration file to use"
    )

    # patch traning simclr
    # parser.add_argument(
    #     "--config",
    #     nargs="?",
    #     type=str,
    #     default="./graph_research/configs/wsi/patch_enc.yml",
    #     help="Configuration file to use"
    # )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    main(cfg)