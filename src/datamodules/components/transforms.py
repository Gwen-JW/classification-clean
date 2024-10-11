import numpy as np


class Normalizer(object):
    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, data):
        if self.norm_type == 'standardization':
            if self.mean is None or self.std is None:
                # self.mean = np.mean(data, axis=0) # need to check self.mean = np.mean(data, axis=(0,1))
                # self.std = np.std(data, axis=0)
                self.mean = np.mean(data, axis=(0,1))
                self.std = np.std(data, axis=(0,1))
        elif self.norm_type == 'min_max':
            if self.min_val is None or self.max_val is None:
                self.min_val = np.min(data, axis=0)
                self.max_val = np.max(data, axis=0)
        elif self.norm_type == "per_sample_std":
            if self.mean is None or self.std is None:
                self.mean = np.mean(data, axis=1, keepdims=True)
                self.std = np.std(data, axis=1, keepdims=True)
        elif self.norm_type == "per_sample_minmax":
            if self.min_val is None or self.max_val is None:
                self.min_val = np.min(data, axis=1, keepdims=True)
                self.max_val = np.max(data, axis=1, keepdims=True)
        else:
            raise (NameError(f'Normalization type {self.norm_type} is not supported.'))
        

    def transform(self, data):
        if self.norm_type == 'standardization':
            return (data - self.mean) / (self.std + np.finfo(float).eps)
        elif self.norm_type == 'min_max':
            return (data - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)
        elif self.norm_type == "per_sample_std":
            return (data - self.mean) / (self.std + np.finfo(float).eps)
        elif self.norm_type == "per_sample_minmax":
            return (data - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)
        else:
            raise (NameError(f'Normalization type {self.norm_type} is not supported.'))

    def inverse_transform(self, data):
        if self.norm_type == 'standardization':
            return data * self.std + self.mean
        elif self.norm_type == 'min_max':
            return data * (self.max_val - self.min_val) + self.min_val
        elif self.norm_type == "per_sample_std":
            return data * self.std + self.mean
        elif self.norm_type == "per_sample_minmax":
            return data * (self.max_val - self.min_val) + self.min_val
        else:
            raise (NameError(f'Normalization type {self.norm_type} is not supported.'))
