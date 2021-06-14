import torch
import random

class RandomChoice(torch.nn.Module):
    def __init__(self, list_of_transforms):
        super().__init__()
        self.list_of_transforms = list_of_transforms

    def forward(self, img):
        transform = random.choice(self.list_of_transforms)
        return transform(img)
