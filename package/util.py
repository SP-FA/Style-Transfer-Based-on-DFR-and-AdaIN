import os
from PIL import Image
import numpy as np
from torch.utils import data


def get_images(imgFile: str) -> dict:
    images = {}
    imgPath = os.listdir(imgFile)
    for i in imgPath:
        text = os.path.splitext(i)
        images[text[0]] = Image.open(imgFile + '/' + i)
    return images


def InfiniteSampler(n):
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31
