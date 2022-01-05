
# in use
from .classic import  Embedder, Identity
from .seam import WSISS


poolings = {
    # wsi methods
    'identity'     : Identity,
    'patch_embed'  : Embedder,

    # SingleStageWSS
    'wsiss'       : WSISS,
}


def load_pooling(pooling, in_channels, num_classes, embed):
    pooling_module = poolings[pooling](in_channels, num_classes, embed)
    return pooling_module