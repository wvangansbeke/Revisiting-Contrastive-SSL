import random
import utils.dataset as datasets

from PIL import ImageFilter

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MultiCropDataset(datasets.ImageFolder):
    def __init__(self, data_path, transform):
        super(MultiCropDataset, self).__init__(data_path)
        self.trans = transform

    def __getitem__(self, index):
        path = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        return multi_crops
