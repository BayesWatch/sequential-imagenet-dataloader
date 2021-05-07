# loads imagenet and writes it into one massive binary file

import os
import numpy as np
from tensorpack.dataflow import dataset, MultiProcessRunnerZMQ, LMDBSerializer
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
    if not os.path.exists(os.environ['TENSORPACK_DATASET']):
        os.mkdir(os.environ['TENSORPACK_DATASET'])

    for name in ['train', 'val']:
        db_filename = 'ILSVRC-%s.lmdb'%name
        db_loc = os.path.join(lmdb_path, db_filename)
        print(f"Processing {args.imagenet} {name} to {db_loc}...")
        ds0 = BinaryILSVRC12(args.imagenet, name)
        ds1 = MultiProcessRunnerZMQ(ds0, num_proc=1)
        LMDBSerializer.save(ds1, db_loc)
