import numpy as np
import os

from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def get_images(imgPath: str) -> dict:
    images = {}
    imgFiles = os.listdir(imgPath)
    for i in imgFiles:
        text = os.path.splitext(i)
        images[text[0]] = Image.open(imgPath + '/' + i)
    return images


def get_iter(dataset, batch, threads):
    DL = DataLoader(dataset, batch_size=batch, sampler=SamplerWrapper(dataset), num_workers=threads)
    return iter(DL)


class SamplerWrapper(Sampler):
    def __init__(self, data):
        self.num = len(data)

    def __iter__(self):
        return iter(self.getSampler())

    def __len__(self):
        return 2 ** 31

    def getSampler(self):
        i = self.num - 1
        order = np.random.permutation(self.num)
        while True:
            yield order[i]
            i += 1
            if i >= self.num:
                np.random.seed()
                order = np.random.permutation(self.num)
                i = 0


class getDataset(Dataset):
    def __init__(self, dataPath, transform):
        dataFiles = os.listdir(dataPath)
        self.transform = transform
        self.dataList = []
        for i in dataFiles:
            transImg = self.transform(Image.open(dataPath + "/" + i))
            self.dataList.append(transImg)

    def __getitem__(self, index):
        return self.dataList[index]

    def __len__(self):
        return len(self.dataList)
