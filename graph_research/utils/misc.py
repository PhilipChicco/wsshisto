
import torch
import numpy as np
import logging
import datetime
import os, time

import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, classification_report
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, confusion_matrix
from collections import OrderedDict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.values = []
        self.counter = 0

    def append(self, val: float) -> None:
        self.values.append(val)
        self.counter += 1

    def val(self) -> float:
        return self.values[-1]

    def avg(self) -> float:
        return sum(self.values) / len(self.values)

    def last_avg(self) -> float:
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state

    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def print_network(net, show_net=False):
    """ Print network definition"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net) if show_net else print("")
    num_params = num_params / 1000000.0
    print("----------------------------")
    print("MODEL: {:.5f}M".format(num_params))
    print("----------------------------")

def get_logger(logdir,name='graph_research'):
    logger = logging.getLogger(name)
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'log_{}.txt'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

## Result utils
def get_metrics(y_label, y_pred, y_probs, title=" ",
                labels=['MSS','MSI'], savepath="./"):

    f1   = f1_score(y_label, y_pred, labels=np.unique(y_label), average='weighted')
    prec = precision_score(y_label, y_pred, average='weighted')
    rec  = recall_score(y_label, y_pred, average='weighted')
    acc  = accuracy_score(y_label, y_pred)
    auc  = roc_auc_score(y_label, y_probs)

    cm1      = confusion_matrix(y_label, y_pred)
    sensitiv = cm1[0,0]/(cm1[0,0] + cm1[0,1])
    spec     = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

    fig, ax = plot_confusion_matrix(conf_mat=np.array(cm1),colorbar=False,show_absolute=True,
                                    show_normed=True,class_names=labels,
                                    figsize=(8.0,8.0))
    plt.title(title)
    plt.savefig(savepath, dpi=300)
    plt.close()
    plt.clf()

    return { title: {
        'f1': f1, 'prec': prec, 'rec': rec, 'spec': spec, 'sens': sensitiv,
        'acc': acc, 'auc': auc
    }}

def find_index(seq, item):
    for i, x in enumerate(seq):
        if item == x:
            return i
    return -1

def adjust_lr_staircase(param_groups, base_lrs, ep, decay_at_epochs=[1, 2], factor=0.1):
    """Multiplied by a factor at the BEGINNING of specified epochs. Different
    param groups specify their own base learning rates.

    Args:
      param_groups: a list of params
      base_lrs: starting learning rates, len(base_lrs) = len(param_groups)
      ep: current epoch, ep >= 1
      decay_at_epochs: a list or tuple; learning rates are multiplied by a factor
        at the BEGINNING of these epochs
      factor: a number in range (0, 1)

    Example:
      base_lrs = [0.1, 0.01]
      decay_at_epochs = [51, 101]
      factor = 0.1
      It means the learning rate starts at 0.1 for 1st param group
      (0.01 for 2nd param group) and is multiplied by 0.1 at the
      BEGINNING of the 51'st epoch, and then further multiplied by 0.1 at the
      BEGINNING of the 101'st epoch, then stays unchanged till the end of
      training.

    NOTE:
      It is meant to be called at the BEGINNING of an epoch.
    """
    assert len(base_lrs) == len(param_groups), \
        "You should specify base lr for each param group."
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep not in decay_at_epochs:
        return

    ind = find_index(decay_at_epochs, ep)

    for i, (g, base_lr) in enumerate(zip(param_groups, base_lrs)):
        g['lr'] = base_lr * factor ** (ind + 1)
        print('===> Param group {}: lr adjusted to {:.10f}'.format(i, g['lr']).rstrip('0'))

    
        
    