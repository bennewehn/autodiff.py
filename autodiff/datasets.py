import numpy as np
import gzip
import struct
import os
from .utils import download


def fetch_mnist() -> tuple[list[tuple[np.ndarray, np.uint8]], list[tuple[np.ndarray, np.uint8]]]:
    """
    Loads the MNIST-Dataset.

    Returns:
        Tuple[list[tuple[np.ndarray, np.uint8]], list[tuple[np.ndarray, np.uint8]]]: tuple
        with training data and test data. Consists of list of tuples with image and label.
    """
    lns = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
           "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

    url = "http://yann.lecun.com/exdb/mnist/"
    dir = "./data/"

    DATA_TYPES = {0x08: np.ubyte,
                  0x09: np.byte,
                  0x0b: np.short,
                  0x0c: np.int32,
                  0x0d: np.float32,
                  0x0e: np.double}

    temp = []
    for ln in lns:
        get(url, ln, dir)
        path = os.path.join(dir, ln)
        with gzip.open(path, 'rb') as f:
            header = f.read(4)
            zeros, data_type, num_dimensions = struct.unpack('>HBB', header)
            data_type = DATA_TYPES[data_type]
            dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                            f.read(4 * num_dimensions))

            data = np.frombuffer(f.read(), dtype=data_type)
            data = data.byteswap()

            # reshape images to single dim
            if len(dimension_sizes) == 3:
                data = data.reshape(
                    dimension_sizes[0], dimension_sizes[1] * dimension_sizes[2])
            else:
                data.reshape(dimension_sizes)

            temp.append(data)

    return list(zip(temp[0], temp[1])), list(zip(temp[2], temp[3]))


def get(url: str, name: str, dir: str) -> None:
    """
    Downloads file by url and name and saves it in dir with name
    if it doesn't exists.

    Args:
        url (str): Url prefix of file to download.
        name (str): Name of file. Gets appended to url.
        dir (str): Directory in which the file should be saved.
    """
    import os
    if not os.path.exists(dir):
        os.mkdir(dir)
    path = os.path.join(dir + name)
    if not os.path.isfile(path):
        download(url + name, path)
