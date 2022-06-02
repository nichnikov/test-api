import numpy as np
from itertools import groupby
from scipy.sparse import hstack, vstack


def pairwise_sparse_jaccard_distance(X, Y=None):
    """
    Computes the Jaccard distance between two sparse matrices or between all pairs in
    one sparse matrix.

    Args:
        X (scipy.sparse.csr_matrix): A sparse matrix.
        Y (scipy.sparse.csr_matrix, optional): A sparse matrix.

    Returns:
        numpy.ndarray: A similarity matrix.
    """

    if Y is None:
        Y = X

    assert X.shape[1] == Y.shape[1]

    X = X.astype(bool).astype(int)
    Y = Y.astype(bool).astype(int)

    intersect = X.dot(Y.T)

    x_sum = X.sum(axis=1).A1
    y_sum = Y.sum(axis=1).A1
    xx, yy = np.meshgrid(x_sum, y_sum)
    union = ((xx + yy).T - intersect)

    return (1 - intersect / union).A


class Searcher:
    """Add data, delete data, storage data, Searching by storage data."""
    def __init__(self):
        self.ids = []
        self.texts = []
        self.matrix = None

    def add(self, ids_: [], texts_: [], vectors_: []):
        """Adding data"""
        # ids_, vectors_ = zip(*ids_vectors)
        self.ids += ids_
        self.texts += texts_
        if self.matrix is None:
            self.matrix = vstack([v.T for v in vectors_])
        else:
            new_matrix = vstack([v.T for v in vectors_])
            self.matrix = vstack([self.matrix, new_matrix])

    def delete(self, ids: []):
        """Deleting data by ids"""
        i_v_t = [(i, v, t) for i, v, t in zip(self.ids, self.matrix, self.texts) if i not in ids]
        if i_v_t:
            ids_, vectors_, texts_ = zip(*i_v_t)
            self.ids = list(ids_)
            self.texts = list(texts_)
            self.matrix = vstack(vectors_)
        else:
            self.ids = []
            self.texts = []
            self.matrix = None

    def search(self, vectors: [], score=0.3):
        """Searching in storage data."""
        searched_matrix = hstack(vectors).T
        jaccard_matrix = 1 - pairwise_sparse_jaccard_distance(searched_matrix, self.matrix)
        indexes = (jaccard_matrix > score).nonzero()
        results = [(i, self.ids[j], self.texts[j], jaccard_matrix[i][j]) for i, j in zip(indexes[0], indexes[1])]
        sorted_results = sorted(results, key=lambda x: x[0])
        grouped_results = [(k, [x for x in v]) for k, v in groupby(sorted_results, key=lambda x: x[0])]
        # return sorted(results, key=lambda x: x[3], reverse=True)
        return [[{"id": x[1], "text": x[2], "score": x[3]} for x in sorted(y[1], key=lambda k: k[3], reverse=True)]
                for y in grouped_results]