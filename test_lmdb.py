
import os
import numpy as np
from tensorpack.dataflow import *
from tensorpack import imgaug

if __name__ == '__main__':
    ds = LMDBData(os.path.join(os.environ['IMAGENET'],'ILSVRC-train.lmdb'), shuffle=False)
    ds = BatchData(ds, 256, use_list=True)
    print(dir(ds))
    print(ds.size())
    for x in ds.get_data():
        print(x)
        assert False
    TestDataSpeed(ds).start()
