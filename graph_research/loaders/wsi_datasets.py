
import os
import cv2
import sys
import numpy as np
import random
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from PIL import Image, ImageEnhance
Image.MAX_IMAGE_PIXELS = 1000000000

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from loaders.utils_wsi import HistoNormalize


class GridWSIPatchDataset(Dataset):
    """
    Data producer that generate all the square grids, e.g. 3x3, of patches,
    from a WSI and its tissue mask, and their corresponding indices with
    respect to the tissue mask
    """

    def __init__(self,
                 wsi_path=None,
                 mask_path=None,
                 image_size=256,
                 patch_size=256,
                 crop_size=256,
                 normalize=True,
                 flip='NONE',
                 rotate='NONE',
                 transform=None,
                 level=0,
                 stride=0.5):
        """
        Initialize the data producer.
        Arguments:
            wsi_path: string, path to WSI file
            mask_path: string, path to mask file in numpy format
            image_size: int, size of the image before splitting into grid, e.g.
                768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
            flip: string, 'NONE' or 'FLIP_LEFT_RIGHT' indicating the flip type
            rotate: string, 'NONE' or 'ROTATE_90' or 'ROTATE_180' or
                'ROTATE_270', indicating the rotate type
        """
        self._wsi_path = wsi_path
        self._mask_path = mask_path
        self._image_size = image_size
        self._patch_size = patch_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self.transform = transform
        self.slide_level = level
        self.n_classes = 2
        self.patch_norm = HistoNormalize()
        self.stride = stride
        self._preprocess()

    def _preprocess(self):

        self._mask = self._mask_path
        self._slide = self._wsi_path

        X_slide, Y_slide = self._slide.level_dimensions[0]
        X_mask, Y_mask = self._mask.shape

        # default
        self._mask = cv2.resize(self._mask.copy(
        ), (self._patch_size, self._patch_size), interpolation=cv2.INTER_NEAREST)
        X_mask, Y_mask = self._mask.shape

        # using large masks
        #X_mask, Y_mask = self._mask.shape
        #self._mask     = cv2.resize(self._mask.copy(), (X_mask//2, Y_mask//2), interpolation=cv2.INTER_NEAREST)
        #X_mask, Y_mask = self._mask.shape
        ##

        self._resolution_x = int(X_slide // X_mask)
        self._resolution_y = int(Y_slide // Y_mask)
        print(
            f'--- Slide {X_slide}, {Y_slide}, {X_mask}, {Y_mask} |res: [{self._resolution_x} |{self._resolution_y}]')
        ##

        self._X_idcs, self._Y_idcs = np.where(self._mask)
        self._idcs_num = len(self._X_idcs)

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        x_mask, y_mask = self._X_idcs[idx], self._Y_idcs[idx]

        # this must be the striode (0.5)
        # try increasting to 1 to avoid overlapping
        x_center = int((x_mask + self.stride) * self._resolution_x)
        y_center = int((y_mask + self.stride) * self._resolution_y)

        m = int(2 ** self.slide_level)
        x = int(x_center - (self._image_size * m) / 2)
        y = int(y_center - (self._image_size * m) / 2)

        img = self._slide.read_region(
            (x, y), self.slide_level, (self._image_size, self._image_size)).convert('RGB')
        img = self.patch_norm(img)
        img = img.resize((256, 256), Image.LANCZOS)

        # print(img.size, self.slide_level, self._image_size)
        # plt.imshow(img)
        # plt.pause(2.00)
        # plt.clf()

        if self._flip == 'FLIP_LEFT_RIGHT':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self._rotate == 'ROTATE_90':
            img = img.transpose(Image.ROTATE_90)

        if self._rotate == 'ROTATE_180':
            img = img.transpose(Image.ROTATE_180)

        if self._rotate == 'ROTATE_270':
            img = img.transpose(Image.ROTATE_270)

        img = self.transform(img)

        return (img, x_mask, y_mask)


class WSIFolders(Dataset):

    def __init__(self,
                 root=None,
                 split='val',
                 transform=None,
                 class_map={'normal': 0, 'tumor': 1},
                 nslides=-1,
                 train=True):

        self.classmap = class_map
        self.nslides = nslides
        self.split = split
        self.root = root
        self.train = train
        lib = os.path.join(root, split + '_lib.pth')

        if not os.path.exists(lib):
            """ Format
               root/val/ ...slide1/ patches ...
               root/train/ ... slide_x/ patches ...

            """
            print('Preprocessing folders .... ')
            lib = self.preprocess()
        elif os.path.isfile(lib):
            print('Using pre-processed lib with patches')
            lib = torch.load(lib)

        else:
            raise ('Please provide root folder or library file')

        self.slidenames = lib['slides']
        self.slides = lib['slides']
        self.grid = []
        self.slideIDX = []
        self.slideLBL = []
        self.targets = lib['targets']

        for idx, (slide, g) in enumerate(zip(lib['slides'], lib['grid'])):
            sys.stdout.write(
                'Opening Slides : [{}/{}]\r'.format(idx + 1, len(lib['slides'])))
            sys.stdout.flush()
            self.grid.extend(g)
            self.slideIDX.extend([idx] * len(g))
            self.slideLBL.extend([self.targets[idx]] * len(g))
        print('')
        print(np.unique(self.slideLBL), len(self.slideLBL), len(self.grid))
        print('Number of tiles: {}'.format(len(self.grid)))

        self.transform = transform

    def __getitem__(self, index):

        slideIDX = self.slideIDX[index]
        target = self.targets[slideIDX]
        img = Image.open(os.path.join(
            self.slides[slideIDX], self.grid[index])).convert('RGB')

        img_i = self.transform['orig'](img)
        img_j = self.transform['aug'](img)

        return img_i, img_j

    def __len__(self):
        return len(self.grid)

    def preprocess(self):
        """
            Change format of lib file to:
            {
                'slides': [xx.tif,xx2.tif , ....],
                'grid'  : [[(x,y),(x,y),..], [(x,y),(x,y),..] , ....],
                'targets': [0,1,0,1,0,1,0, etc]
            }
            len(slides) == len(grid) == len(targets)
        """
        grid = []
        targets = []
        slides = []
        class_names = [str(x) for x in range(len(self.classmap))]
        for i, cls_id in enumerate(class_names):
            slide_dicts = os.listdir(
                os.path.join(self.root, self.split, cls_id))
            print('--> | ', cls_id, ' | ', len(slide_dicts))
            for idx, slide in enumerate(slide_dicts[:self.nslides]):
                slide_folder = os.path.join(
                    self.root, self.split, cls_id, slide)
                grid_number = len(os.listdir(slide_folder))
                # skip empty folder
                if grid_number == 0:
                    print("Skipped : ", slide, cls_id, ' | ', grid_number)
                    continue

                grid_p = []
                for id_patch in os.listdir(slide_folder):
                    grid_p.append(id_patch)

                if not slide_folder in slides:
                    slides.append(slide_folder)
                    grid.append(grid_p)
                    targets.append(int(cls_id))

        print(len(slides), len(grid), len(targets))
        return {'slides': slides, 'grid': grid, 'targets': targets}


class SlideCompressed(Dataset):

    def __init__(self,
                 root=None,
                 split='val',
                 transform=None,
                 class_map={'normal': 0, 'tumor': 1},
                 nslides=-1,
                 train=False,
                 use_json=True,
                 use_multilab=False,
                 mask_dir=None):

        self.root = root
        self.split = split
        self.nslides = nslides
        self.transfrom = transform
        self.classmap = class_map
        self.train = train
        self.use_json = use_json
        self.use_multilab = use_multilab
        self.mask_dir = mask_dir


        if self.train == True:
            print(f'Using Augmentations ........')

        self.rot_choices = [0, 90, 180, 270]
        self.flip_choices = ['none', 'vertical', 'horizontal', 'both']

        self.slides, self.slides_mask, self.targets = self.preprocess()
        print(f'TOTAL SLIDES : {len(self.slides)}')

    def rot_flip_array(self, array, axes, rot_deg, flip):
        """
        Batch augmentation function supporting 90 degree rotations and flipping.
        Args:
            array: batch in [b, x, y, c] format.
            axes: axes to apply the transformation.
            rot_deg (int): rotation degree (0, 90, 180 or 270).
            flip (str): flipping augmentation ('none', 'horizontal', 'vertical' or 'both'.
        Returns: batch array.
        """

        # Rot
        array = self.aug_rot(array, degrees=rot_deg, axes=axes)

        # Flip
        if flip == 'vertical':
            array = np.flip(array, axis=axes[0])
        elif flip == 'horizontal':
            array = np.flip(array, axis=axes[1])
        elif flip == 'both':
            array = np.flip(array, axis=axes[0])
            array = np.flip(array, axis=axes[1])
        elif flip == 'none':
            pass

        return array

    def aug_rot(self, array, degrees, axes):
        """
        90 degree rotation.
        Args:
            array: batch in [b, x, y, c] format.
            degrees (int): rotation degree (0, 90, 180 or 270).
            axes: axes to apply the transformation.
        Returns: batch array.
        """

        if degrees == 0:
            pass
        elif degrees == 90:
            array = np.rot90(array, k=1, axes=axes)
        elif degrees == 180:
            array = np.rot90(array, k=2, axes=axes)
        elif degrees == 270:
            array = np.rot90(array, k=3, axes=axes)

        return array

    def __getitem__(self, index):

        slide_object = self.slides[index]
        slide_label = self.targets[index]
        slide_mask = self.slides_mask[index]

        slide_name = os.path.split(slide_object)[-1]
        slide_name = slide_name.split('.')[0]

        slide_object = np.load(slide_object).astype(np.float32)
        slide_mask = np.load(slide_mask).astype(np.float32)/255.0
        #
        slide_mask[slide_mask > 0] = 1.0

        if self.train == True:
            rot_flip_index = random.randint(0, 3)
            r_i = self.rot_choices[rot_flip_index]
            flip_i = self.flip_choices[rot_flip_index]
            slide_object = self.rot_flip_array(
                slide_object.copy(), [0, 1], r_i, flip_i).astype(np.float32)
            slide_mask = self.rot_flip_array(
                slide_mask.copy(), [0, 1], r_i, flip_i).astype(np.float32)

        if self.use_multilab:
            slide_cls = slide_label
            slide_label = np.zeros((2), np.float32)
            slide_label[slide_cls] = 1.0

        if self.transfrom is not None:
            slide_object = self.transfrom(slide_object)
            slide_mask   = torch.from_numpy(slide_mask).long()

        return slide_object, slide_label, slide_mask, slide_name

    def __len__(self):
        return len(self.targets)

    def preprocess(self):
        """
            Collect compressed slides from folder
        """
        targets = []
        slides = []
        slides_mask = []
        self.slide_paths = []

        if self.use_json:
            split_map = {'train': 'base.lib',
                         'val': 'val.lib', 'test': 'novel.lib'}
            lib = os.path.join(self.root, split_map[self.split])
            lib = torch.load(lib)
            # lib has :
            # image_paths | image_names | image_label
            if self.mask_dir is not None:
                print(f'Using Pseudo-CAM masks | {self.mask_dir}')
            # image_names have full paths to slides or npy

            for idx, (slide, cls_id, wsi_path) in enumerate(zip(lib['image_paths'], 
                lib['image_label'], lib['image_names'])):
                slide_name = os.path.split(slide)[-1]

                if self.mask_dir is None:
                    slide_mask = slide.replace(
                        '/wsi/{}'.format(slide_name), '/labels/{}'.format(slide_name))
                else:
                    slide = os.path.join(self.mask_dir, 'wsi', slide_name)
                    slide_mask = os.path.join(
                        self.mask_dir, 'labels', slide_name)

                if not slide in slides:
                    self.slide_paths.append(wsi_path)
                    slides.append(slide)
                    slides_mask.append(slide_mask)
                    targets.append(int(cls_id))
            
            # use specific number of slides
            if not self.nslides == -1 and self.nslides < len(slides):
                print('Using SubSet of Slides ....',self.nslides, ' per class :::::')
                new_targets = []
                new_slides_mask = []
                new_slides = []
                
                for clc_idx in range(len(self.classmap)):
                    t_list =  (np.where(np.array(targets) == clc_idx)[0]).tolist()
                    new_targets += list(np.array(targets)[t_list])[:self.nslides]
                    new_slides += list(np.array(slides)[t_list])[:self.nslides]
                    new_slides_mask += list(np.array(slides_mask)[t_list])[:self.nslides]
            
                slides = new_slides
                slides_mask = new_slides_mask
                targets = new_targets


        else:
            class_names = [str(x) for x in range(len(self.classmap))]
            for i, cls_id in enumerate(class_names):
                slide_dicts = os.listdir(os.path.join(
                    self.root, self.split, 'wsi', cls_id))
                print('--> | ', cls_id, ' | ', len(slide_dicts))
                for idx, slide in enumerate(slide_dicts[:self.nslides]):
                    slide_folder = os.path.join(
                        self.root, self.split, 'wsi', cls_id, slide)
                    slide_folder_labels = os.path.join(
                        self.root, self.split, 'labels', cls_id, slide)
                    slides.append(slide_folder)
                    slides_mask.append(slide_folder_labels)
                    targets.append(int(cls_id))

        print(len(slides), len(slides_mask), len(targets))
        return slides, slides_mask, targets


