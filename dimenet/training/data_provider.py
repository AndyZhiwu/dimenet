from collections import OrderedDict
import numpy as np
import torch
from .data_container import index_keys


class DataProvider:
    def __init__(self, data_container, ntrain, nvalid, batch_size=1,
                 seed=None, randomized=False):
        self.data_container = data_container
        self._ndata = len(data_container)
        self.nsamples = {'train': ntrain, 'val': nvalid, 'test': len(data_container) - ntrain - nvalid}
        self.batch_size = batch_size

        # Random state parameter, such that random operations are reproducible if wanted
        self._random_state = np.random.RandomState(seed=seed)

        all_idx = np.arange(len(self.data_container))
        if randomized:
            # Shuffle indices
            all_idx = self._random_state.permutation(all_idx)

        # Store indices of training, validation and test data
        self.idx = {'train': all_idx[0:ntrain],
                    'val': all_idx[ntrain:ntrain+nvalid],
                    'test': all_idx[ntrain+nvalid:]}

        # Index for retrieving batches
        self.idx_in_epoch = {'train': 0, 'val': 0, 'test': 0}

        # dtypes of dataset values
        self.dtypes_input = OrderedDict()
        self.dtypes_input['Z'] = torch.int32
        self.dtypes_input['R'] = torch.float32
        for key in index_keys:
            self.dtypes_input[key] = torch.int32
        self.dtype_target = torch.float32

        # Shapes of dataset values
        self.shapes_input = {}
        self.shapes_input['Z'] = [None]
        self.shapes_input['R'] = [None, 3]
        for key in index_keys:
            self.shapes_input[key] = [None]
        self.shape_target = [None, len(data_container.target_keys)]

    def shuffle_train(self):
        """Shuffle the training data"""
        self.idx['train'] = self._random_state.permutation(self.idx['train'])

    def get_batch_idx(self, split):
        """Return the indices for a batch of samples from the specified set"""
        start = self.idx_in_epoch[split]

        # Is epoch finished?
        if self.idx_in_epoch[split] == self.nsamples[split]:
            start = 0
            self.idx_in_epoch[split] = 0

        # shuffle training set at start of epoch
        if start == 0 and split == 'train':
            self.shuffle_train()

        # Set end of batch
        self.idx_in_epoch[split] += self.batch_size
        if self.idx_in_epoch[split] > self.nsamples[split]:
            self.idx_in_epoch[split] = self.nsamples[split]
        end = self.idx_in_epoch[split]

        return self.idx[split][start:end]

    def idx_to_data(self, idx, device='cpu'):
        """Convert a batch of indices to a batch of data"""
        batch = self.data_container[idx]

        inputs = {}
        for key in self.dtypes_input.keys():
            if key in batch:
                inputs[key] = torch.tensor(batch[key], dtype=self.dtypes_input[key], device=device)
        targets = torch.tensor(batch['targets'], dtype=torch.float32, device=device)
        return (inputs, targets)

    def get_batch(self, split, device='cpu'):
        """Get a batch of data"""
        idx = self.get_batch_idx(split)
        return self.idx_to_data(idx, device=device)
