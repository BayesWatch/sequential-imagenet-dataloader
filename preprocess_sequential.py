# loads imagenet and writes it into one massive binary file

import os
import numpy as np
from tensorpack.dataflow import dataset, PrefetchDataZMQ, LMDBSerializer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess ImageNet into an LMDB file.')
    parser.add_argument('imagenet', type=str, help='location of ImageNet directory')
    parser.add_argument('database_dir', type=str, default=None, help='location to save output database')
    args = parser.parse_args()

    class BinaryILSVRC12(dataset.ILSVRC12Files):
        def get_data(self):
            for fname, label in super(BinaryILSVRC12, self).__iter__():
                with open(fname, 'rb') as f:
                    jpeg = f.read()
                jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
                yield [jpeg, label]
    if args.database_dir is None:
        lmdb_path = args.imagenet
    else:
        lmdb_path = args.database_dir 
    os.environ['TENSORPACK_DATASET'] = os.path.join(lmdb_path, "tensorpack_data")

    for name in ['train', 'val']:
        print(f"Processing {args.imagenet} {name}...")
        ds0 = BinaryILSVRC12(args.imagenet, name)
        ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
        LMDBSerializer.save(ds1, os.path.join(lmdb_path, 'ILSVRC-%s.lmdb'%name))
