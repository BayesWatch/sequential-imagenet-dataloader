
A plug-in ImageNet DataLoader for PyTorch. Uses Tensorpack DataFlow's
[sequential loading][seq] to load fast even if you're using a HDD. 

[seq]: http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html#sequential-read

Install
-------

Requirements:

* [DataFlow][]: clone and `pip install -e .`

   ```
   sudo apt-get install build-essential libcap-dev
   pip install python-prctl
   pip install --upgrade git+https://github.com/tensorpack/dataflow.git
   ```

* [LMDB][]: `pip install lmdb`

* [TQDM][]: `pip install tqdm`

* [OpenCV][]: `pip install opencv-python` (if you install opencv with conda it's
likely to be slow due to being compiled with openmp, if you're not sure
your opencv is fast enough check using the script
[here](https://github.com/tensorpack/benchmarks/blob/master/ImageNet/benchmark-opencv-resize.py))

* [Protobuf][]: `conda install protobuf`

[tensorpack]: https://github.com/ppwwyyxx/tensorpack
[lmdb]: https://lmdb.readthedocs.io/en/release/
[tqdm]: https://pypi.python.org/pypi/tqdm
[opencv]: https://pypi.python.org/pypi/opencv-python
[Protobuf]: https://github.com/google/protobuf

If you use pip's editable install, you can [tune the speed of the DataLoader
on your system][tune] by modifying this code.

[tune]: https://tensorpack.readthedocs.io/en/latest/tutorial/performance-tuning.html

```
git clone https://github.com/BayesWatch/sequential-imagenet-dataloader.git
cd sequential-imagenet-dataloader
pip install -e .
```

Or install directly:

```
pip install git+https://github.com/BayesWatch/sequential-imagenet-dataloader.git
```

### Preprocessing

Before being able to train anything, you have to run the preprocessing
script `preprocess_sequential.py` to create the LMDB binary files.  They
will get put in the directory specified and they will take up 140G for
train, plus more for val. Use the script with these arguments:

```
python preprocess_sequential.py <imagenet directory> <directory to save lmdb files>
```

Usage
-----

Wherever the `DataLoader` is defined in your Pytorch code, replaced that
with `imagenet_seq.data.Loader`; although you can't call it with exactly
the same arguments. For an example, this would be the substitution in the
[PyTorch ImageNet example][imagenet]:

```
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
imagenet_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

# train_dataset = datasets.ImageFolder(traindir, imagenet_transforms)
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
#    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

train_loader = ImagenetLoader(args.data, 'train', imagenet_transforms,
        batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
```

For a complete example, see the [example ImageNet training
script provided][example].

[example]: ./examples/imagenet/main.py

Experiments
-----------

Running the example ImageNet script on a workstation with a single Titan X
using 4 workers.

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  TITAN Xp            Off  | 00000000:17:00.0 Off |                  N/A |
| 52%   80C    P2    85W / 250W |   8837MiB / 12196MiB |    100%      Default |
|                               |                      |                  N/A |
```

Comparing estimated hours to completion, each with 4 threads dedicated to
workers loading the dataset:

* Shared SSD NFS: 
    * Loading from folders of images: 85 hours
* HDD:
    * Loading from folders of images: 670 hours
    * Loading from LMDB as DataFlow: 70 hours

Estimated times neglect the validation set. Real completion time for the
experiment using LMDB as DataFlow with the code in this repository
completed in 75 hours.

To check that this still converges to the benchmark accuracy (the shuffling
is only local so may not match), I ran the experiment until completion
using this DataLoader. The final validation top-1 accuracy was 69.76% and
the charts detailing this experiment can be found [here][wandbreport].

[wandbreport]: https://wandb.ai/gngdb/trial-imagenet/reports/Training-Resnet18-with-sequential-imagenet-dataloader--Vmlldzo3MTI3MzY
