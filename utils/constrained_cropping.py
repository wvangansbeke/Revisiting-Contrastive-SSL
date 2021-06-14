"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import math
import random
import warnings
import torch.utils.data as data


from PIL import Image
from torchvision.transforms import functional as F


class CustomMultiCropDataset(data.Dataset):
    def __init__(self, base_dataset, multi_crop, secondary_transform,
                return_crop_params=True):
        self.base_dataset = base_dataset
        self.multi_crop = multi_crop
        assert(self.base_dataset.transform is None)
        if isinstance(secondary_transform, list):
            assert(len(secondary_transform) == self.multi_crop.N_large + self.multi_crop.N_small)
        self.secondary_transform = secondary_transform
        self.return_crop_params = return_crop_params

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample = self.base_dataset.__getitem__(index)
        multi_crop, multi_crop_params = self.multi_crop(sample)
        assert(len(multi_crop) == self.multi_crop.N_large + self.multi_crop.N_small)

        if isinstance(self.secondary_transform, list):
            multi_crop = [tf(x) for tf, x in zip(self.secondary_transform, multi_crop)]

        else:
            multi_crop = [self.secondary_transform(x) for x in multi_crop]

        if self.return_crop_params:
            return multi_crop, multi_crop_params

        else:
            return multi_crop

        
def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


def _compute_intersection(box1, box2):
    i1, j1, h1, w1 = box1
    i2, j2, h2, w2 = box2
    x_overlap = max(0, min(j1+w1, j2+w2) - max(j1, j2))
    y_overlap = max(0, min(i1+h1, i2+h2) - max(i1, i2))
    return x_overlap * y_overlap
   

class CustomMultiCropping(object):
    """ This class implements a custom multi-cropping strategy. In particular, 
    we generate the following crops:

    - N_large random crops of random size (default: 0.2 to 1.0) of the orginal size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. The crops
    are finally resized to the given size (default: 160). 

    - N_small random crops of random size (default: 0.05 to 0.14) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. The crops
    are finally resized to the given size (default: 96). There is the possibility to condition
    the smaller crops on the last large crop. Note that the latter is used as the key for MoCo.

    Args:
        size_large: expected output size for large crops
        scale_large: range of size of the origin size cropped for large crops
        
        size_small: expected output size for small crops
        scale_small: range of size of the origin size cropped for small crops

        N_large: number of large crops
        N_small: number of small crops
        
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR

        condition_small_crops_on_key: condition small crops on key
    """
    def __init__(self, size_large=160, scale_large=(0.2, 1.0), 
                    size_small=96, scale_small=(0.05, 0.14), N_large=2, N_small=4, 
                    ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR,
                    condition_small_crops_on_key=True):
        if isinstance(size_large, (tuple, list)):
            self.size_large = size_large
        else:
            self.size_large = (size_large, size_large)
        
        if isinstance(size_small, (tuple, list)):
            self.size_small = size_small
        else:
            self.size_small = (size_small, size_small)

        if (scale_large[0] > scale_large[1]) or (scale_small[0] > scale_small[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation

        self.scale_large = scale_large
        self.scale_small = scale_small

        self.N_large = N_large
        self.N_small = N_small

        self.ratio = ratio
        self.condition_small_crops_on_key = condition_small_crops_on_key

    @staticmethod
    def get_params(img, scale, ratio, ):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def get_params_conditioned(self, img, scale, ratio, constraint):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
            constraint (tuple): params (i, j, h, w) that should be used to constrain the crop

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width
        for counter in range(10):
            rand_scale = random.uniform(*scale)
            target_area = rand_scale * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                intersection = _compute_intersection((i, j, h, w), constraint)
                if intersection >= 0.1 * target_area: # 10 percent of the small crop is part of big crop.
                    return i, j, h, w
        
        return self.get_params(img, scale, ratio) # Fallback to default option

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            multi_crop (list of lists): result of multi-crop
        """
        multi_crop = []
        multi_crop_params = []
        for ii in range(self.N_large):
            i, j, h, w = self.get_params(img, self.scale_large, self.ratio)
            multi_crop_params.append((i, j, h, w))
            multi_crop.append(F.resized_crop(img, i, j, h, w, self.size_large, self.interpolation))

        for ii in range(self.N_small):
            if not self.condition_small_crops_on_key:
                i, j, h, w = self.get_params(img, self.scale_small, self.ratio)

            else:
                i, j, h, w = self.get_params_conditioned(img, self.scale_small, self.ratio,
                                                            multi_crop_params[self.N_large -1])
                
            multi_crop_params.append((i, j, h, w))
            multi_crop.append(F.resized_crop(img, i, j, h, w, self.size_small, self.interpolation))

        return multi_crop, multi_crop_params 

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size_large={0}'.format(self.size_large)
        format_string += ', scale_large={0}'.format(tuple(round(s, 4) for s in self.scale_large))
        format_string += ', size_small={0}'.format(tuple(round(s, 4) for s in self.size_small))
        format_string += ', scale_small={0}'.format(tuple(round(s, 4) for s in self.scale_small))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', condition_small_crops_on_key={})'.format(self.condition_small_crops_on_key)
        return format_string
