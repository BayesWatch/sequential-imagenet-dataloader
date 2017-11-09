
import os
import numpy as np
from tensorpack.dataflow import *

if __name__ == '__main__':
    imagenet_loc = os.environ['IMAGENET']
    ds0 = dataset.ILSVRC12(imagenet_loc, 'train', shuffle=True)
    ds1 = BatchData(ds0, 256, use_list=True)
    TestDataSpeed(ds1).start()

