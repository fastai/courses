import numpy as np
import bcolz
import threading

class BcolzArrayIterator(object):
    """
    Returns an iterator object into Bcolz carray files
    Original version by Thiago Ramon Gonçalves Montoya
    Docs (and discovery) by @MPJansen
    Refactoring, performance improvements, fixes by Jeremy Howard j@fast.ai
        :Example:
        X = bcolz.open('file_path/feature_file.bc', mode='r')
        y = bcolz.open('file_path/label_file.bc', mode='r')
        trn_batches = BcolzArrayIterator(X, y, batch_size=64, shuffle=True)
        model.fit_generator(generator=trn_batches, samples_per_epoch=trn_batches.N, nb_epoch=1)
        :param X: Input features
        :param y: (optional) Input labels
        :param w: (optional) Input feature weights
        :param batch_size: (optional) Batch size, defaults to 32
        :param shuffle: (optional) Shuffle batches, defaults to false
        :param seed: (optional) Provide a seed to shuffle, defaults to a random seed
        :rtype: BcolzArrayIterator
        >>> A = np.random.random((32*10 + 17, 10, 10))
        >>> c = bcolz.carray(A, rootdir='test.bc', mode='w', expectedlen=A.shape[0], chunklen=16)
        >>> c.flush()
        >>> Bc = bcolz.open('test.bc')
        >>> bc_it = BcolzArrayIterator(Bc, shuffle=True)
        >>> C_list = [next(bc_it) for i in range(11)]
        >>> C = np.concatenate(C_list)
        >>> np.allclose(sorted(A.flatten()), sorted(C.flatten()))
        True
    """

    def __init__(self, X, y=None, w=None, batch_size=32, shuffle=False, seed=None):
        if y is not None and len(X) != len(y):
            raise ValueError('X (features) and y (labels) should have the same length'
                             'Found: X.shape = %s, y.shape = %s' % (X.shape, y.shape))
        if w is not None and len(X) != len(w):
            raise ValueError('X (features) and w (weights) should have the same length'
                             'Found: X.shape = %s, w.shape = %s' % (X.shape, w.shape))
        if batch_size % X.chunklen != 0:
            raise ValueError('batch_size needs to be a multiple of X.chunklen')

        self.chunks_per_batch = batch_size // X.chunklen
        self.X = X
        self.y = y if y is not None else None
        self.w = w if w is not None else None
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.shuffle = shuffle
        self.seed = seed


    def reset(self): self.batch_index = 0


    def next(self):
        with self.lock:
            if self.batch_index == 0:
                if self.seed is not None:
                    np.random.seed(self.seed + self.total_batches_seen)
                self.index_array = (np.random.permutation(self.X.nchunks + 1) if self.shuffle
                    else np.arange(self.X.nchunks + 1))

            #batches_x = np.zeros((self.batch_size,)+self.X.shape[1:])
            batches_x, batches_y, batches_w = [],[],[]
            for i in range(self.chunks_per_batch):
                current_index = self.index_array[self.batch_index]
                if current_index == self.X.nchunks:
                    batches_x.append(self.X.leftover_array[:self.X.leftover_elements])
                    current_batch_size = self.X.leftover_elements
                else:
                    batches_x.append(self.X.chunks[current_index][:])
                    current_batch_size = self.X.chunklen
                self.batch_index += 1
                self.total_batches_seen += 1

                idx = current_index * self.X.chunklen
                if not self.y is None: batches_y.append(self.y[idx: idx + current_batch_size])
                if not self.w is None: batches_w.append(self.w[idx: idx + current_batch_size])
                if self.batch_index >= len(self.index_array):
                    self.batch_index = 0
                    break

            batch_x = np.concatenate(batches_x)
            if self.y is None: return batch_x

            batch_y = np.concatenate(batches_y)
            if self.w is None: return batch_x, batch_y

            batch_w = np.concatenate(batches_w)
            return batch_x, batch_y, batch_w


    def __iter__(self): return self

    def __next__(self, *args, **kwargs): return self.next(*args, **kwargs)

