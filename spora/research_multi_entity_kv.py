"""Validation helpers for experimental combined key/value streams."""
import torch


def concatenate_kv(first, second, *, reverse=False):
    if type(reverse) is not bool or len(first)!=2 or len(second)!=2:
        raise ValueError('Expected two key/value pairs')
    a,b=first,second
    if (any(not isinstance(t,torch.Tensor) or t.ndim!=2 for t in (*a,*b))
            or a[0].shape[0]!=a[1].shape[0] or b[0].shape[0]!=b[1].shape[0]
            or not a[0].shape[0] or not b[0].shape[0]
            or a[0].shape[1]!=b[0].shape[1] or a[1].shape[1]!=b[1].shape[1]):
        raise ValueError('Incompatible key/value pairs')
    if reverse:a,b=b,a
    return torch.cat((a[0],b[0]),0),torch.cat((a[1],b[1]),0)
