from skimage.segmentation import watershed
from skimage.draw import circle
from skimage.feature import peak_local_max
import scipy.ndimage as ndi
import numpy as np

from skimage.morphology import remove_small_objects
from scipy.spatial import distance as dist
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from skimage.morphology import dilation, erosion
from scipy.signal import convolve2d
from PIL import Image


class Watershed():
    def __init__(self, mask_th, multichannel=False, ch=None):
        self.mask_th = mask_th
        self.multichannel = multichannel
        self.ch = ch

    def __call__(self, img):
        if self.multichannel:
            img = img[:, :, self.ch]
        mask = (img > self.mask_th)
        markers = get_markers(img)
        markers = ndi.label(markers)[0]
        seg = watershed(image=-img, markers=markers, mask=mask)
        return seg


def get_markers(prob_map):
    shape = prob_map.shape
    local_max = np.zeros(shape)
    index_local_max = peak_local_max(prob_map, 1, 0.01)
    for r, c in index_local_max:
        rr, cc = circle(r, c, 5, shape)
        local_max[rr, cc] = 1
    label_local_max = ndi.label(local_max)[0]
    unique_max = np.unique(label_local_max)[1:]
    index_merged_max = ndi.measurements.center_of_mass(local_max,
                                                       labels=label_local_max,
                                                       index=unique_max)
    index_merged_max = np.round(index_merged_max).astype(np.int)
    merged_max = np.zeros(shape)
    rr = index_merged_max[:, 0]
    cc = index_merged_max[:, 1]
    merged_max[rr, cc] = 1
    return merged_max


def aji_score(label, pred):
    """
    AJI as described in the paper, but a much faster implementation.
    Peter Naylor paper with a fix by Miguel Luna
    """
    G = label.astype(np.int)
    S = pred.astype(np.int)
    if S.sum() == 0:
        return 0.
    C = 0
    U = 0
    USED = np.zeros(S.max())
    G_flat = G.flatten()
    S_flat = S.flatten()
    G_max = np.max(G_flat)
    S_max = np.max(S_flat)
    m_labels = max(G_max, S_max) + 1
    cm = confusion_matrix(G_flat, S_flat,
                          labels=list(range(m_labels))).astype(np.float)
    LIGNE_J = np.zeros(S_max)
    for j in range(1, S_max + 1):
        LIGNE_J[j - 1] = cm[:, j].sum()

    for i in range(1, G_max + 1):
        LIGNE_I_sum = cm[i, :].sum()

        def h(indice):
            LIGNE_J_sum = LIGNE_J[indice - 1]
            inter = cm[i, indice]
            union = LIGNE_I_sum + LIGNE_J_sum - inter
            return inter / union

        def inter(indice):
            inter = cm[i, indice]
            return inter

        JI_ligne_i = np.array(list(map(inter, range(1, S_max + 1))))
        if JI_ligne_i.sum() == 0:
            U += cm[i, 0]
        else:
            JI_ligne = np.array(list(map(h, range(1, S_max + 1))))
            best_indice = np.argmax(JI_ligne) + 1
            C += cm[i, best_indice]
            U += LIGNE_J[best_indice - 1] + LIGNE_I_sum - cm[i, best_indice]
            USED[best_indice - 1] = 1

    U_sum = ((1 - USED) * LIGNE_J).sum()
    U += U_sum
    return float(C) / float(U)


def all_scores(label, pred):
    label = np.round(label)
    pred  = np.round(pred)
    label = np.clip(label, 0, 1)
    pred  = np.clip(pred, 0, 1)
    label = np.reshape(label, (-1))
    pred  = np.reshape(pred, (-1))
    dice = f1_score(label, pred)
    iou  = intersection_over_union(pred, label)
    acc  = accuracy_score(pred, label)
    return dice, iou, acc

def get_scores_mil(label, pred):
    dice, iou, acc = all_scores(label, pred)
    return dice, iou, acc

def get_scores(label, pred):
    #pred = remove_small_objects(pred)
    #aji  = aji_score(label, pred)
    dice, iou, acc = all_scores(label, pred)
    #iou  = intersection_over_union(pred, label)
    #acc  = accuracy_score(pred, label)
    return dice, iou, acc


def get_inner_boundary_mask(img):
    se = np.ones((3, 3))
    ero = img.copy()
    ero[ero == 0] = ero.max() + 1
    ero = erosion(ero, se)
    ero[img == 0] = 0
    grad = dilation(img, se) - ero
    grad[img == 0] = 0
    grad[grad > 0] = 1
    return grad


def get_boundary(array):
    kernel = np.ones((3, 3))
    f_array = array.astype(np.float)
    c_array = convolve2d(f_array, kernel, mode="same", boundary="symm") / 9
    b_array = np.not_equal(f_array, c_array) * 9
    b_array *= np.clip(array, 0, 1)
    return (b_array).astype(np.float)


def overlap_preds(img, label):
    boundary = get_boundary(label).astype(np.bool)
    img[:, :, 0][boundary] = 2
    img[:, :, 1][boundary] = 2
    img = Image.fromarray(np.uint8(img * 255))  # .convert('RGB')
    return img

def numeric_score(prediction, groundtruth):
    """Computation of statistical numerical scores:
    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives
    return: tuple (FP, FN, TP, TN)
    """
    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    return FP, FN, TP, TN

def intersection_over_union(prediction, groundtruth):

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP + FN) <= 0.0:
        return 0.0
    return TP / (TP + FP + FN) #* 100.0

def accuracy_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy #* 100.0