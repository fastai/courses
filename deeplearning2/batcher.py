from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings

class Iterator(object):

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

class Batcher(Iterator):
    def __init__(self, x, y, batch_size=64, shuffle=False, seed=None, proc_fn=None):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))
        self.x = np.asarray(x)
        self.y = np.asarray(y) if y is not None else None
        self.proc_fn = proc_fn
        super(Batcher, self).__init__(x.shape[0], batch_size, shuffle, seed)


    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), self.x.dtype)
        for i, j in enumerate(index_array):
            x = self.x[j]
            #x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        res = batch_x if self.y is None else batch_x, self.y[index_array]
        if self.proc_fn: res=self.proc_fn(res)
        return res

