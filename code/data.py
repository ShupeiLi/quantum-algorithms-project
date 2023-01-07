# -*- coding: utf-8 -*-

from sklearn import datasets
import numpy as np
from sklearn.decomposition import TruncatedSVD
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

    zero_data_tr, zero_label_tr = data_split(0, 100)
    zero_data_te, zero_label_te = data_split(100, 110)
    one_data_tr, one_label_tr = data_split(177 + 1, 177 + 101)
    one_data_te, one_label_te = data_split(177 + 101, 177 + 111)

    data_tr = np.vstack((zero_data_tr, one_data_tr))
    label_tr = np.hstack((zero_label_tr, one_label_tr))
    data_te = np.vstack((zero_data_te, one_data_te))
    label_te = np.hstack((zero_label_te, one_label_te))

    # Dimension reduction
    assert method == 'svd' or method == 'umap', f'Invalid method parameter {method}.'
    if method == 'svd':
        fitter = TruncatedSVD(n_components=n_components)
    else:
        fitter = umap.UMAP(n_components=n_components)
    data_tr = fitter.fit_transform(data_tr)
    data_te = fitter.fit_transform(data_te)
    data_tr = data_tr / np.abs(data_tr).max()
    data_te = data_te / np.abs(data_te).max()
    return {'train': {'X': data_tr, 'y': label_tr},
            'test': {'X': data_te, 'y': label_te}}
