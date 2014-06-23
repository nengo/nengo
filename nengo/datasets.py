import os
import urllib

from nengo.env import cache_dir

dataset_dir = os.path.join(cache_dir, 'datasets')


def full_dataset_path(filename):
    return os.path.join(dataset_dir, filename)


def fetch_file(url, destname, force=False):
    """Fetch a dataset file from a URL.

    Parameters
    ----------
    url : string
        The URL where the file is located.
    dest : string
        Relative path to the destination.
    force : bool, optional
        If True, download the dataset even if the destination file exists.
    """
    path = full_dataset_path(destname)
    if not os.path.exists(path) or force:
        # ensure directory exists
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # download dataset
        urllib.urlretrieve(url, filename=path)


def load_mnist(key):
    """Load the MNIST handwritten digit dataset.

    Parameters
    ----------
    key : {'train', 'test', 'valid'}
        'train':
          Load the training data, N = 50 000 images. In the literature,
          the 60 000-image training set is the 'train' and 'valid' sets.
        'test':
          Load the testing data, N = 10 000 images.
        'valid':
          Load the validation data, N = 10 000 images.

    Returns
    -------
    images : (N, 784) ndarray
        Array of images; each row is one flattened 28 x 28 image
        with intensity from 0 to 1.
    labels : (N,) ndarray
        Array of labels; each row is an integer label from 0 to 9.

    """
    import gzip
    import cPickle as pickle

    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = os.path.join('mnist', 'mnist.pkl.gz')
    keys = ['train', 'valid', 'test']
    if key not in keys:
        raise ValueError("Unrecognized dataset '%s'" % key)

    # download the data
    fetch_file(url, filename)

    # load data from file
    path = full_dataset_path(filename)
    with gzip.open(path, 'rb') as f:
        setlist = pickle.load(f)

    # return the requested data
    sets = dict(zip(keys, setlist))
    return sets[key]
