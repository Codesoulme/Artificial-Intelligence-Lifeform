"""Ordered identity-token vector comparison, only for controlled research."""
import math
import torch
import torch.nn.functional as F


def ordered_similarity(query, record):
    if (not isinstance(query,torch.Tensor) or not isinstance(record,torch.Tensor)
            or query.ndim!=2 or record.ndim!=2 or query.shape[1]!=record.shape[1]
            or not query.shape[0] or not record.shape[0]
            or not bool(torch.isfinite(query).all() and torch.isfinite(record).all())):
        raise ValueError('Invalid token representations')
    if bool((query.norm(dim=-1)==0).any() or (record.norm(dim=-1)==0).any()):
        raise ValueError('Zero token representation')
    if query.shape[0]!=record.shape[0]:
        return -1.0
    q=F.normalize(query.float(),dim=-1);r=F.normalize(record.float(),dim=-1)
    return float((q*r).sum(-1).min())


def select_ordered(query, records, threshold, tie_tolerance=1e-6):
    if (not isinstance(records,(list,tuple)) or not records
            or isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not math.isfinite(threshold)
            or isinstance(tie_tolerance,bool) or not isinstance(tie_tolerance,(int,float))
            or not math.isfinite(tie_tolerance) or tie_tolerance<0):
        raise ValueError('Invalid selection inputs')
    scores=[ordered_similarity(query,record) for record in records]
    order=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)
    ambiguous=len(order)>1 and scores[order[0]]-scores[order[1]]<=tie_tolerance
    return {'index':order[0] if scores[order[0]]>threshold and not ambiguous else None,
            'scores':scores,'ambiguous':ambiguous}
