import numpy as np
import bcolz
import threading

class BcolzArrayIterator(object):
    """
    Returns an iterator object into Bcolz carray files
    Thanks to R4mon: http://pastebin.com/y0DskiK6
        :Example:

        X = bcolz.open('file_path/feature_file.bc', mode='r')
        y = bcolz.open('file_path/label_file.bc', mode='r')
        trn_batches = BcolzArrayIterator(X, y, batch_size=X.chunklen * batch_size, shuffle=True)
        model.fit_generator(generator=trn_batches, samples_per_epoch=trn_batches.N, nb_epoch=1)

        :param X: Input features
        :param y: (optional) Input labels
        :param w: (optional) Input feature weights
        :param batch_size: (optional) Batch size, defaults to 32
        :param shuffle: (optional) Shuffle batches, defaults to false
        :param seed: (optional) Provide a seed to shuffle, defaults to a random seed
        :rtype: BcolzArrayIterator

        >>> A = np.random.random((32*100 + 16, 100, 100))
        >>> c = bcolz.carray(A, rootdir='test.bc', mode='w', expectedlen=A.shape[0])
        >>> c.flush()
        >>> Bc = bcolz.open('test.bc')
        >>> bc_it = BcolzArrayIterator(Bc, batch_size=Bc.chunklen, shuffle=False)
        >>> C_list = [v for v in bc_it]
        >>> C = np.concatenate(C_list)
        >>> (A == C).all()
        True
    """

    def __init__(self, X, y=None, w=None, batch_size=32, shuffle=False, seed=None):
        if y is not None and len(X) != len(y):
            raise ValueError('X (features) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' % (X.shape, y.shape))
        if w is not None and len(X) != len(w):
            raise ValueError('X (features) and w (weights) '
                             'should have the same length. '
                             'Found: X.shape = %s, w.shape = %s' % (X.shape, w.shape))
        self.X = X
        if y is not None:
            self.y = y[:]
        else:
            self.y = None
        if w is not None:
            self.w = w[:]
        else:
            self.w = None
        self.N = X.shape[0]
        self.batch_index = 0
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.shuffle = shuffle
        self.seed = seed
        self.index_array = np.arange(self.N)

        self.reset()

    def reset(self):
        self.batch_index = 0

        if self.seed is not None:
            np.random.seed(self.seed)
        if self.shuffle:
            self.index_array = np.random.permutation(self.N)
        else:
            self.index_array = np.arange(self.N)

    def next(self):
        if self.batch_index == self.N:
            raise StopIteration
        with self.lock:
            batch_start = self.batch_index
            batch_end = min(batch_start + self.batch_size, self.N)
            selected_idx = self.index_array[batch_start:batch_end]
            self.batch_index = batch_end

            batch_x = self.X[selected_idx]
            if self.y is None:
                return batch_x

            batch_y = self.y[selected_idx]
            if self.w is None:
                return batch_x, batch_y

            batch_w = self.w[selected_idx]
            return batch_x, batch_y, batch_w

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
