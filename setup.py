"""Based on example https://github.com/pypa/sampleproject"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pytorch-imagenet-dataloader',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.1',

    description='A faster PyTorch ImageNet loader.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/gngdb/pytorch-imagenet-dataloader',

    # Author details
    author='Gavin Gray',
    author_email='gavingray1729@gmail.com',

    # Choose your license
    license='MIT',

    # What does your project relate to?
    keywords='pytorch',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['lmdb', 'tqdm'],
)
