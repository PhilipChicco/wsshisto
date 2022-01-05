from .wsi_seg_eval import WSISegTest
from .wsi_ss_eval import WSISSTest


testers_dict = {
    # seg base
    'wsiseg' : WSISegTest,
    'wsiss'  : WSISSTest,
}

def get_tester(name):
    names = list(testers_dict.keys())
    if name not in names:
        raise ValueError('Invalid choice for evaluators - choices: {}'.format(' | '.join(names)))
    return testers_dict[name]