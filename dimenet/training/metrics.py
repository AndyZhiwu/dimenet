import numpy as np
import torch


class Metrics:
    def __init__(self, tag, targets, ex=None):
        self.tag = tag
        self.targets = targets
        self.ex = ex

        self.loss_sum = 0.0
        self.loss_count = 0
        self.mean_mae_sum = 0.0
        self.mean_mae_count = 0
        self.maes_sum = np.zeros(len(targets))
        self.maes_count = 0

    def update_state(self, loss, mean_mae, mae, nsamples):
        self.loss_sum += loss * nsamples
        self.loss_count += nsamples
        self.mean_mae_sum += mean_mae * nsamples
        self.mean_mae_count += nsamples
        self.maes_sum += mae * nsamples
        self.maes_count += nsamples

    def reset_states(self):
        self.loss_sum = 0.0
        self.loss_count = 0
        self.mean_mae_sum = 0.0
        self.mean_mae_count = 0
        self.maes_sum = np.zeros(len(self.targets))
        self.maes_count = 0

    def keys(self):
        keys = [f'loss_{self.tag}', f'mean_mae_{self.tag}', f'mean_log_mae_{self.tag}']
        keys.extend([key + '_' + self.tag for key in self.targets])
        return keys

    def result(self):
        result_dict = {}
        result_dict[f'loss_{self.tag}'] = self.loss
        result_dict[f'mean_mae_{self.tag}'] = self.mean_mae
        result_dict[f'mean_log_mae_{self.tag}'] = self.mean_log_mae
        for i, key in enumerate(self.targets):
            result_dict[key + '_' + self.tag] = self.maes[i].item()
        return result_dict

    @property
    def loss(self):
        if self.loss_count > 0:
            return (self.loss_sum / self.loss_count).item() if isinstance(self.loss_sum, torch.Tensor) else self.loss_sum / self.loss_count
        return 0.0

    @property
    def maes(self):
        if self.maes_count > 0:
            return self.maes_sum / self.maes_count
        return self.maes_sum

    @property
    def mean_mae(self):
        if self.mean_mae_count > 0:
            return (self.mean_mae_sum / self.mean_mae_count).item() if isinstance(self.mean_mae_sum, torch.Tensor) else self.mean_mae_sum / self.mean_mae_count
        return 0.0

    @property
    def mean_log_mae(self):
        maes = self.maes
        if self.maes_count > 0:
            return np.mean(np.log(maes + 1e-10)).item()
        return 0.0
