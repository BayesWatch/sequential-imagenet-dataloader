# dataloader respecting the PyTorch conventions, but using tensorpack to load and process
# includes typical augmentations for ImageNet training

import os

import cv2
import torch

import numpy as np
import tensorpack.dataflow as td
from tensorpack import imgaug
from tensorpack.utils.serialize import loads

##############################################################################################################
# copied from: https://github.com/tensorpack/tensorpack/blob/master/examples/ImageNetModels/imagenet_utils.py#
############################################################################################################## 
def fbresnet_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    interpolation = cv2.INTER_CUBIC
    # linear seems to have more stable performance.
    # but we keep cubic for compatibility with old models
    if isTrain:
        augmentors = [
            imgaug.GoogleNetRandomCropAndResize(interp=interpolation),
            imgaug.ToFloat32(),  # avoid frequent casting in each color augmentation
            # It's OK to remove the following augs if your CPU is not fast enough.
            # Removing brightness/contrast/saturation does not have a significant effect on accuracy.
            # Removing lighting leads to a tiny drop in accuracy.
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4)),
                 imgaug.Contrast((0.6, 1.4), rgb=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.ToUint8(),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, interp=interpolation),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors
#####################################################################################################
#####################################################################################################


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

    def __init__(self, imagenet_dir, mode, batch_size=256, shuffle=False, num_workers=4, cache=50000,
            drop_last=False, cuda=False):
        if drop_last:
            raise NotImplementedError("drop_last not implemented")
        # enumerate standard imagenet augmentors
        assert mode in ['train', 'val'], mode
        augmentors = fbresnet_augmentor(mode == 'train')

        # load the lmdb if we can find it
        lmdb_loc = os.path.join(imagenet_dir, 'ILSVRC-%s.lmdb'%mode)
        ds = td.LMDBData(lmdb_loc, shuffle=False)
        ds = td.MapData(ds, td.LMDBSerializer._deserialize_lmdb) 
        ds = td.MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
        # ds = td.LMDBSerializer.load(lmdb_loc, shuffle=False)
        # ds = td.LocallyShuffleData(ds, cache)
        # ds = td.PrefetchData(ds, batch_size*num_workers, 1)
        # ds = td.MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
        # ds = td.MapDataComponent(ds, lambda x: loads(x), 0)
        # ds = td.AugmentImageComponent(ds, imagenet_augmentors)
        # https://github.com/tensorpack/tensorpack/issues/1405
        # def f(x):
        #     img, label = td.LMDBSerializer._deserialize_lmdb(x)
        #     return imagenet_augmentors.augment(cv2.imdecode(img, cv2.IMREAD_COLOR)), label
        # def g(x):
        #     return imagenet_augmentors.augment(cv2.imdecode(x, cv2.IMREAD_COLOR))
        ds = td.AugmentImageComponent(ds, augmentors, copy=False)
        ds = td.MultiProcessRunnerZMQ(ds, num_proc=num_workers)
        # ds = td.MultiProcessMapDataZMQ(ds, num_proc=num_workers, map_func=f)
        self.ds = td.BatchData(ds, batch_size, remainder=False)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cuda = cuda

    def __iter__(self):
        for x, y in self.ds.get_data():
            if self.cuda:
                # images come out as uint8, which are faster to copy onto the gpu
                x = torch.ByteTensor(x).cuda()
                y = torch.IntTensor(y).cuda()
                # but once they're on the gpu, we'll need them in float
                yield uint8_to_float(x), y.long()
            else:
                yield uint8_to_float(torch.ByteTensor(x)), torch.IntTensor(y).long()

    def __len__(self):
        return self.ds.size()

def uint8_to_float(x):
    x = x.permute(0,3,1,2) # pytorch is (n,c,w,h)
    return x.float()/128. - 1.

if __name__ == '__main__':
    from tqdm import tqdm
    dl = ImagenetLoader(os.environ['DBLOC'], 'train', cuda=True, num_workers=4)
    for x in tqdm(dl, total=len(dl)):
        pass
