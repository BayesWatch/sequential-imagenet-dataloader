
A plug-in ImageNet DataLoader for PyTorch. Uses tensorpack's [sequential
loading][seq] to load fast even if you're using a HDD. 

[seq]: http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html#sequential-read

Install
-------

Requirements:

* [Tensorpack][]: clone and `pip install -e .`
* [LMDB][]: `pip install lmdb`
* [TQDM][]: `pip install tqdm`
* [OpenCV][]: `conda install opencv`
* [Protobuf][]: `conda install protobuf`

[tensorpack]: https://github.com/ppwwyyxx/tensorpack
[lmdb]: https://lmdb.readthedocs.io/en/release/
[tqdm]: https://pypi.python.org/pypi/tqdm
[opencv]: https://pypi.python.org/pypi/opencv-python
[Protobuf]: https://github.com/google/protobuf

If you use pip's editable install, you can fix bugs I have probably introduced:

```
git clone https://github.com/BayesWatch/sequential-imagenet-dataloader.git
cd sequential-imagenet-dataloader
pip install -e .
```

To start, you must set the environment variable `IMAGENET` to point to
wherever you have saved the ILSVRC2012 dataset. You must also set the
`TENSORPACK_DATASET` environment variable, because tensorpack may download
some things itself.

### Preprocessing

Before being able to train anything, you have to run the preprocessing
script `preprocess_sequential.py` to create the huge LMDB binary files.
They will get put in wherever your `IMAGENET` environment variable is, and
they will take up 140G for train, plus more for val.

Usage
-----

Wherever the `DataLoader` is defined in your Pytorch code, replaced that
with `imagenet_seq.data.Loader`; although you can't call it with exactly
the same arguments. For an example, this would be the substitution in the
[PyTorch ImageNet example][imagenet]:

```
    #train_loader = torch.utils.data.DataLoader(
    #    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #    num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    train_loader = ImagenetLoader('train', batch_size=args.batch_size, num_workers=args.workers)
```

You may need to tune the number of workers to use to get best results.

Experiments
-----------

Running the [PyTorch ImageNet Example][imagenet] on the server I work on
that has no SSD, but a set of 4 Titan X GPUs, I get an average
minibatch speed of 5.3s. Using this iterator to feed examples, I'm able to
get about 0.59s per minibatch, so 54 minutes per epoch; 90 epochs should
take about 73 hours, and that's enough to get results. A resnet-18
converged to 69% top-1 and 89% top-5, which [appears to be the
standard][resnet_original].

The Titan Xs still look a little hungry if we're running on all four, but
it's fast enough to work with.

[imagenet]: https://github.com/pytorch/examples/tree/master/imagenet
[resnet_original]: https://github.com/HolmesShuan/ResNet-18-Caffemodel-on-ImageNet
