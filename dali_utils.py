from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types

import pdb

class ExternalInputIterator(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        self.i = 0
        self.n = len(self.dataset)

        return self

    def __next__(self):
        
        batch = []
        labels = []
        for _ in range(self.batch_size):
            img, lab = self.dataset.getitem_dali(self.i)
            self.i = (self.i + 1) % self.n

            batch.append(img)
            labels.append(lab)


        return (batch, labels)

class ExternalInputCallable(object):
    def __init__(self, dataset):
        self.dataset = dataset

        self.full_iterations = len(self.dataset) // 2

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration()

        img, lab = self.dataset.getitem_dali(sample_idx)

        return img, lab

