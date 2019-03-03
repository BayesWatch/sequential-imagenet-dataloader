# loads imagenet and writes it into one massive binary file

import os
import numpy as np
from tensorpack.dataflow import *
os.environ['TENSORPACK_DATASET'] = '/home/shgao/tensorpack_data/'
os.environ['IMAGENET'] = '/media/db/datasets/ILSVRC/Data/CLS-LOC/'
if __name__ == '__main__':
    class BinaryILSVRC12(dataset.ILSVRC12Files):
        def get_data(self):
            for fname, label in super(BinaryILSVRC12, self).__iter__():
                with open(fname, 'rb') as f:
                    jpeg = f.read()
                jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
                yield [jpeg, label]
    imagenet_path = os.environ['IMAGENET']

    for name in ['train', 'val']:
        ds0 = BinaryILSVRC12(imagenet_path, name)
        ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
        LMDBSerializer.save(ds1, os.path.join(imagenet_path,'ILSVRC-%s.lmdb'%name))
