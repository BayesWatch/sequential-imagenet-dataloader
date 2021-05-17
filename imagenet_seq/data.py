# dataloader respecting the PyTorch conventions, but using tensorpack to load and process

import os

import cv2
import torch

import numpy as np
import dataflow as td
from io import BytesIO
from PIL import Image


class ImagenetLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int): how many samples per batch to load
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 4)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
    """

    def __init__(self, imagenet_dir, mode, transform, batch_size, shuffle=False, num_workers=4, cache=50000,
            drop_last=False):
        if drop_last:
            raise NotImplementedError("drop_last not implemented")
        # enumerate standard imagenet augmentors
        assert mode in ['train', 'val'], mode

        # open the lmdb file
        lmdb_loc = os.path.join(imagenet_dir, 'ILSVRC-%s.lmdb'%mode)
        ds = td.LMDBData(lmdb_loc, shuffle=False)
        if shuffle:
            ds = td.LocallyShuffleData(ds, cache)
        def f(x):
            img, label= td.LMDBSerializer._deserialize_lmdb(x)
            # img, label = x
            img = Image.open(BytesIO(img.tobytes())).convert('RGB')
            img = transform(img)
            return img, label
        # ds = td.MultiProcessMapDataZMQ(ds, num_proc=num_workers, map_func=f)
        ds = td.MultiThreadMapData(ds, num_thread=num_workers, map_func=f)
        # ds = td.MapData(ds, f)
        self.ds = td.BatchData(ds, batch_size, use_list=True, remainder=False)
        # self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.ds.reset_state()
        self.ds_iter = iter(self.ds)
        self.N = self.ds.size()
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if (self.i + 1) == self.N:
            raise StopIteration
        x, y = next(self.ds_iter)
        self.i += 1
        x, y = torch.stack(x), torch.tensor(y)
        return x, y

    def __len__(self):
        return self.N

if __name__ == '__main__':
    from tqdm import tqdm
    import torchvision.transforms as transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    dl = ImagenetLoader(os.environ['DBLOC'], 'train', transform, 256, num_workers=4, shuffle=True)
    # td.TestDataSpeed(dl.ds).start()
    for x, y in tqdm(dl, total=len(dl)):
        x, y = x.cuda(), y.cuda()

