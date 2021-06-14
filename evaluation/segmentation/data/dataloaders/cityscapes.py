#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import cv2
import json
import numpy as np
import os
import zipfile

from collections import namedtuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import verify_str_arg
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from data.util.mypath import Path


class Cityscapes(Dataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset. """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
        ]

    def __init__(
            self,
            root: str = Path.db_root_dir('cityscapes'),
            split: str = "train",
            use_semseg: bool = True,
            transform: Optional[Callable] = None,
            cities: Optional[list] = None,
    ) -> None:
        super(Cityscapes, self).__init__()
        self.root = root
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.split = split
        self.use_semseg = use_semseg
        self.transform = transform
       
        valid_modes = ("train", "test", "val") 
        verify_str_arg(split, "split", valid_modes)
       
        self.images = []
        self.images_dir = os.path.join(self.root, 'leftImg8bit', self.split)
 
        if self.use_semseg:
            self.semseg_dir = os.path.join(self.root, 'gtFine', self.split)
            self.semseg = []
            
            # id -> train_id
            self.semseg_mapping = {class_[1]: class_[2] for class_ in self.classes}
            self.inverse_semseg_mapping = {255: 255}
            for class_ in self.classes:
                if not class_[2] in [-1,255]:
                    self.inverse_semseg_mapping[class_[2]] = class_[1]
            self.semseg_color = {self.semseg_mapping[class_[1]]: class_[-1] for class_ in self.classes}
                    
        if cities:
            print('A list of cities was provided {} ({})'.format(cities, self.split))

        else:
            cities = os.listdir(self.images_dir)

        for city in cities:
            img_dir = os.path.join(self.images_dir, city)
            if self.use_semseg:
                semseg_dir = os.path.join(self.semseg_dir, city)
            
            for file_name in os.listdir(img_dir):
                img_name = os.path.join(img_dir, file_name)
                assert(os.path.exists(img_name))
                self.images.append(img_name)

                if self.use_semseg:
                    semseg_name = os.path.join(semseg_dir, 
                                    '{}_gtFine_labelIds.png'.format(file_name.split('_leftImg8bit')[0]))
                    assert(os.path.exists(semseg_name))
                    self.semseg.append(semseg_name)
        assert(len(self.semseg) == len(self.images))

    def __getitem__(self, index: int) -> dict:
        sample = {}
        sample['image'] = self._load_img(index)
        sample['meta'] = self._load_meta(index)

        if self.use_semseg:
            sample['semseg'] = self._load_semseg(index)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.images)

    def _load_img(self, index) -> np.array:
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.uint8)
        return _img
        
    def _load_meta(self, index) -> dict:
        # Return location/file.png
        fname_split = self.images[index].split('/')
        meta = {'fname': os.path.join(fname_split[-2], fname_split[-1]), 'db_name': 'cityscapes'}
        return meta

    def _load_semseg(self, index) -> np.array:
        _semseg = np.array(Image.open(self.semseg[index])).astype(np.uint8)
        _tmp = np.zeros_like(_semseg)
        for id_ in np.unique(_semseg):
            _tmp[_semseg == id_] = self.semseg_mapping[id_]

        return _tmp

    def convert_semseg_to_color(self, semseg):
        h, w = semseg.shape
        _tmp = np.zeros((h, w, 3), dtype=np.uint8)
        for id_ in np.unique(semseg):
            mask = (semseg == id_)
            _tmp[mask, 0] = self.semseg_color[id_][0]
            _tmp[mask, 1] = self.semseg_color[id_][1]
            _tmp[mask, 2] = self.semseg_color[id_][2]

        return _tmp

    def convert_to_eval(self, semseg):
        # Convert predictions to format that can be used for eval.
        if not hasattr(self, 'inverse_semseg_mapping'):
            raise ValueError('There is no inverse segmentation mapping that can be used for cityscapesScripts')
        h, w = semseg.shape
        _tmp = np.zeros((h, w), dtype=np.uint8)
        for id_ in np.unique(semseg):
            mask = (semseg == id_)
            _tmp[mask] = self.inverse_semseg_mapping[id_]
        return _tmp

    def get_class_names(self):
        return [class_[0] for class_ in self.classes if class_[2] != 255] 

    def __str__(self):
        return 'Cityscapes(split=' + str(self.split) + ')'


def main():
    """ For purpose of debugging """
    import matplotlib.pyplot as plt
    from utils.common_config import get_train_transformations
    tr = get_train_transformations({'train_db_kwargs': {'transforms': 'vanilla_cityscapes'}})
    dataset = Cityscapes(transform=tr, cities=['zurich', 'stuttgart'])
    print('Number of samples {}'.format(len(dataset))) 
    for i in range(10):
        sample = dataset.__getitem__(i)
        fig, axes = plt.subplots(2)
    
        img = sample['image']
        img[0] = img[0] * 0.229 + 0.485
        img[1] = img[1] * 0.224 + 0.456
        img[2] = img[2] * 0.225 + 0.406
        img = np.transpose(img.numpy(), (1,2,0))
        axes[0].imshow(img)
        axes[1].imshow(dataset.convert_semseg_to_color(sample['semseg'].numpy()))
        plt.show()

if __name__=='__main__':
    main()
