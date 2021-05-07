# dataloader respecting the PyTorch conventions, but using tensorpack to load and process
# includes typical augmentations for ImageNet training

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
        batch_size (int, optional): how many samples per batch to load
            (default: 256).
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
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the 
            GPU, which is faster).
    """

    def __init__(self, imagenet_dir, mode, transform, batch_size=256, shuffle=False, num_workers=4, cache=50000,
            drop_last=False, cuda=False):
        if drop_last:
            raise NotImplementedError("drop_last not implemented")
        # enumerate standard imagenet augmentors
        assert mode in ['train', 'val'], mode

        # load the lmdb if we can find it
        lmdb_loc = os.path.join(imagenet_dir, 'ILSVRC-%s.lmdb'%mode)
        # ds = td.LMDBData(lmdb_loc, shuffle=False)
        # ds = td.MapData(ds, td.LMDBSerializer._deserialize_lmdb) 
        # ds = td.MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
        ds = td.LMDBSerializer.load(lmdb_loc, shuffle=False)
        ds = td.LocallyShuffleData(ds, cache)
        # ds = td.PrefetchData(ds, batch_size*num_workers, 1)
        # ds = td.MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
        # ds = td.MapDataComponent(ds, lambda x: loads(x), 0)
        # ds = td.AugmentImageComponent(ds, imagenet_augmentors)
        # https://github.com/tensorpack/tensorpack/issues/1405
        def f(x):
            img, label = x
            img = Image.open(BytesIO(img.tobytes())).convert('RGB')
            img = transform(img)
            return img, label
        # def g(x):
        #     return imagenet_augmentors.augment(cv2.imdecode(x, cv2.IMREAD_COLOR))
        # ds = td.AugmentImageComponent(ds, augmentors, copy=False)
        # ds = td.MultiProcessRunnerZMQ(ds, num_proc=num_workers)
        # ds = td.MultiProcessMapDataZMQ(ds, num_proc=num_workers, map_func=f)
        ds = td.MapData(ds, f)
        self.ds = td.BatchData(ds, batch_size, use_list=True, remainder=False)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cuda = cuda

    def __iter__(self):
        for x, y in self.ds.get_data():
            x, y = torch.stack(x), torch.tensor(y)
            yield x, y

    def __len__(self):
        return self.ds.size()

def uint8_to_float(x):
    x = x.permute(0,3,1,2) # pytorch is (n,c,w,h)
    return x.float()/128. - 1.

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
    dl = ImagenetLoader(os.environ['DBLOC'], 'train', transform=transform, cuda=True, num_workers=4)
    for x, y in tqdm(dl, total=len(dl)):
        x, y = x.cuda(), y.cuda()
        pass
