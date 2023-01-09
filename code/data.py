# -*- coding: utf-8 -*-

from sklearn import datasets
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
import umap


def digits(method='svd', n_components=2):
    """Load digits dataset in sklearn.

    Args:
        method: 'svd' - Truncated SVD, 'umap' - UMAP. Default: 'svd'.
        n_components: Dimension of the embedded space. Default: 2.
    """
    # Load data
    data = datasets.load_digits()
    def data_split(start_index, end_index):
        dataset = data['data'][data['target'].argsort()][start_index:end_index]
        label = data['target'][data['target'].argsort()][start_index:end_index]
        return dataset, label

    zero_data, zero_label = data_split(0, 160)
    one_data, one_label = data_split(177 + 1, 177 + 161)
    data = np.vstack((zero_data, one_data))
    label = np.hstack((zero_label, one_label))

    # Dimension reduction
    assert method == 'svd' or method == 'umap', f'Invalid method parameter {method}.'
    if method == 'svd':
        fitter = TruncatedSVD(n_components=n_components)
    else:
        fitter = umap.UMAP(n_components=n_components)
    data = fitter.fit_transform(data)
    data = data / np.abs(data).max()
    data = np.concatenate((data, np.expand_dims(label, axis=-1)), axis=1)
    np.random.seed(42)
    np.random.shuffle(data)
    return {'X': data[:, :n_components], 'y': data[:, n_components].astype(int)}


def moons():
    """Load make_moons dataset in sklearn.
    """
    # Load data
    data = datasets.make_moons(n_samples=200, random_state=42)
    label = data[1]
    data = data[0]
    data = data / np.abs(data).max()
    return {'X': data, 'y': label}


def cancer(method='svd', n_components=2):
    """Load breast cancer dataset in sklearn.

    Args:
        method: 'svd' - Truncated SVD, 'umap' - UMAP. Default: 'svd'.
        n_components: Dimension of the embedded space. Default: 2.
    """
    # Load data
    data_arr = datasets.load_breast_cancer()
    data = data_arr['data']
    label = data_arr['target']

    # Dimension reduction
    assert method == 'svd' or method == 'umap', f'Invalid method parameter {method}.'
    if method == 'svd':
        fitter = TruncatedSVD(n_components=n_components)
    else:
        fitter = umap.UMAP(n_components=n_components)
    data = fitter.fit_transform(data)
    data = data / np.abs(data).max()
    return {'X': data, 'y': label}


def cross_validation_split(data, n_folds=5):
    """Prepare data for cross validation.

    Args:
        data: Format {'X': features, 'y': labels}.
        n_folds: The number of folds. Default: 5.
    """
    cv = list()
    kf = KFold(n_splits=n_folds)
    for i, (train_index, test_index) in enumerate(kf.split(data['X'])):
        cv.append({'train': {'X': data['X'][train_index, :],
                             'y': data['y'][train_index]},
                   'test': {'X': data['X'][test_index, :],
                             'y': data['y'][test_index]},
            })
    return cv
