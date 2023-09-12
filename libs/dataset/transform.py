import numpy as np
import torch
import copy

class Normalize(object):
    def __init__(self, sets = ['support', 'query']):
        self.sets = sets
        self.mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3]).astype(np.float32)
        self.std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3]).astype(np.float32)
    def __call__(self, sample):
        for set in self.sets:
            sample[set]['img'] = ((sample[set]['img'] / 255.0 - self.mean) / self.std).astype(np.float32)
        return 


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.
    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """
    def __init__(self, keys=['img', 'mask'], sets=['support', 'query']):
        self.keys = keys
        self.sets = sets

    def __call__(self, sample):
        data = copy.deepcopy(sample)
        for set in self.sets:
            for key in self.keys:
                if len(data[set][key].shape) < 3:
                    data[set][key] = np.expand_dims(data[set][key], -1) #维度扩展
                data[set][key] = to_tensor(data[set][key]).permute(2, 0, 1).contiguous()
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'