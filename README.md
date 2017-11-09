
Requirements:

* [Tensorpack][]: clone and `pip install -e .`
* [LMDB][]: `pip install lmdb`
* [TQDM][]: `pip install tqdm`
* [OpenCV][]: `conda install opencv`

[tensorpack]: 
[lmdb]: 
[tqdm]: 

To start, you must set the environment variable `IMAGENET` to point to
wherever you have saved the ILSVRC2012 dataset. You must also set the
`TENSORPACK_DATASET` environment variable, because tensorpack may download
some things itself.
