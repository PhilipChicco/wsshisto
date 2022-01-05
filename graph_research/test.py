import warnings
warnings.filterwarnings("ignore")

import argparse
import yaml
import os
import os,sys , torch, random, numpy as np

# trainers
from evaluators import get_tester
from utils.misc import get_logger


def main(cfg):
    print(cfg)
    print()

    # setup logdir, writer and logger
    logdir = os.path.join(cfg['root'], cfg['testing']['logdir'])
    os.makedirs(logdir,exist_ok=True)

    tester_name = cfg['evaluator']

    with open(os.path.join(logdir, tester_name + '.yml'), 'w') as fp:
        yaml.dump(cfg, fp)

    logger = get_logger(logdir, name=tester_name)

    Tester = get_tester(tester_name)(cfg, logdir, logger)
    print()

    # start testing
    Tester.test()

if __name__ == '__main__':
    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # get configs
    parser = argparse.ArgumentParser(description="Test a Network")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./graph_research/configs/wsi/wsi_seg_cm.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    main(cfg)