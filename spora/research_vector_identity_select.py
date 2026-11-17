"""Research-only vector-based record selection with strict abstention."""
import math
import torch
import torch.nn.functional as F


def select_identity(query, records, threshold, tie_tolerance=1e-6):
    if (not isinstance(query,torch.Tensor) or not isinstance(records,torch.Tensor)
            or query.ndim!=1 or records.ndim!=2 or records.shape[0]<1
            or records.shape[1]!=query.shape[0] or not query.is_floating_point()
            or not records.is_floating_point() or not bool(torch.isfinite(query).all())
            or not bool(torch.isfinite(records).all())
            or isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not math.isfinite(threshold)
            or not 0<=tie_tolerance<1):
        raise ValueError('Invalid identity selection inputs')
    if float(query.norm())==0 or bool((records.norm(dim=-1)==0).any()):
        raise ValueError('Zero identity representation')
    scores=F.normalize(records.float(),dim=-1) @ F.normalize(query.float(),dim=-1)
    top=torch.topk(scores,min(2,len(scores)))
    winner=int(top.indices[0]);best=float(top.values[0])
    ambiguous=len(scores)>1 and float(top.values[0]-top.values[1])<=tie_tolerance
    return {'index':winner if best>threshold and not ambiguous else None,
            'scores':[float(x) for x in scores],'ambiguous':ambiguous}
