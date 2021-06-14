import math
import torch

from PIL import Image
from torch import Tensor
import utils.auto_augment.functional as F
from utils.auto_augment.functional import InterpolationMode

def _get_transforms():
    # Transforms for ImageNet
    return [
        (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
        (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
        (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
        (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
        (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
        (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
        (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
        (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
        (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
        (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
        (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
        (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
        (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
        (("Invert", 0.6, None), ("Equalize", 1.0, None)),
        (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
        (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
        (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
        (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
        (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
        (("Color", 0.4, 0), ("Equalize", 0.6, None)),
        (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
        (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
        (("Invert", 0.6, None), ("Equalize", 1.0, None)),
        (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
        (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
    ]


def _get_magnitudes():
    # Magnitudes
    _BINS = 10
    return {
        # name: (magnitudes, signed)
        "ShearX": (torch.linspace(0.0, 0.3, _BINS), True),
        "ShearY": (torch.linspace(0.0, 0.3, _BINS), True),
        "TranslateX": (torch.linspace(0.0, 150.0 / 331.0, _BINS), True),
        "TranslateY": (torch.linspace(0.0, 150.0 / 331.0, _BINS), True),
        "Rotate": (torch.linspace(0.0, 30.0, _BINS), True),
        "Brightness": (torch.linspace(0.0, 0.9, _BINS), True),
        "Color": (torch.linspace(0.0, 0.9, _BINS), True),
        "Contrast": (torch.linspace(0.0, 0.9, _BINS), True),
        "Sharpness": (torch.linspace(0.0, 0.9, _BINS), True),
        "Posterize": (torch.tensor([8, 8, 7, 7, 6, 6, 5, 5, 4, 4]), False),
        "Solarize": (torch.linspace(256.0, 0.0, _BINS), False),
        "AutoContrast": (None, None),
        "Equalize": (None, None),
        "Invert": (None, None),
    }


class AutoAugment(torch.nn.Module):
    def __init__(self, interpolation = InterpolationMode.NEAREST, fill=None):
        super().__init__()
        self.interpolation = interpolation
        self.fill = fill
        self.transforms = _get_transforms()
        self._op_meta = _get_magnitudes()

    @staticmethod
    def get_params(transform_num):
        policy_id = torch.randint(transform_num, (1,)).item()
        probs = torch.rand((2,))
        signs = torch.randint(2, (2,))

        return policy_id, probs, signs

    def _get_op_meta(self, name):
        return self._op_meta[name]

    def forward(self, img):
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        transform_id, probs, signs = self.get_params(len(self.transforms))

        for i, (op_name, p, magnitude_id) in enumerate(self.transforms[transform_id]):
            if probs[i] <= p:
                magnitudes, signed = self._get_op_meta(op_name)
                magnitude = float(magnitudes[magnitude_id].item()) \
                    if magnitudes is not None and magnitude_id is not None else 0.0
                if signed is not None and signed and signs[i] == 0:
                    magnitude *= -1.0

                if op_name == "ShearX":
                    img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                                   interpolation=self.interpolation, fill=fill)

                elif op_name == "ShearY":
                    img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                                   interpolation=self.interpolation, fill=fill)

                elif op_name == "TranslateX":
                    img = F.affine(img, angle=0.0, translate=[int(F._get_image_size(img)[0] * magnitude), 0], scale=1.0,
                                   interpolation=self.interpolation, shear=[0.0, 0.0], fill=fill)

                elif op_name == "TranslateY":
                    img = F.affine(img, angle=0.0, translate=[0, int(F._get_image_size(img)[1] * magnitude)], scale=1.0,
                                   interpolation=self.interpolation, shear=[0.0, 0.0], fill=fill)

                elif op_name == "Rotate":
                    img = F.rotate(img, magnitude, interpolation=self.interpolation, fill=fill)

                elif op_name == "Brightness":
                    img = F.adjust_brightness(img, 1.0 + magnitude)

                elif op_name == "Color":
                    img = F.adjust_saturation(img, 1.0 + magnitude)

                elif op_name == "Contrast":
                    img = F.adjust_contrast(img, 1.0 + magnitude)

                elif op_name == "Sharpness":
                    img = F.adjust_sharpness(img, 1.0 + magnitude)

                elif op_name == "Posterize":
                    img = F.posterize(img, int(magnitude))

                elif op_name == "Solarize":
                    img = F.solarize(img, magnitude)

                elif op_name == "AutoContrast":
                    img = F.autocontrast(img)

                elif op_name == "Equalize":
                    img = F.equalize(img)

                elif op_name == "Invert":
                    img = F.invert(img)

                else:
                    raise ValueError("The provided operator {} is not recognized.".format(op_name))

        return img


    def __repr__(self):
        return self.__class__.__name__ 
