"""Parameter-matched linear and nonlinear pair features for offline probes."""
import math
import torch


def question_rows(rows):
    lookup = {}
    for i, row in enumerate(rows):
        if row['variant'] == 'current_only':
            if row['case'] in lookup:
                raise ValueError('Duplicate question row')
            lookup[row['case']] = i
    indices = []
    for row in rows:
        if row['case'] not in lookup:
            raise ValueError('Missing question row')
        index = lookup[row['case']]
        source = rows[index]
        if (source['group'] != row['group'] or
                source['current_message_id'] != row['current_message_id']):
            raise ValueError('Question source mismatch')
        indices.append(index)
    return indices


def pair_features(context, question, mode):
    if context.ndim != 2 or context.shape != question.shape or context.shape[1] == 0:
        raise ValueError('Expected aligned nonempty matrices')
    if not bool(torch.isfinite(context).all() and torch.isfinite(question).all()):
        raise ValueError('Nonfinite input')
    if mode == 'linear_pair':
        extra = ((question + context) / math.sqrt(2), (question - context) / math.sqrt(2))
    elif mode == 'interaction_pair':
        extra = (question * context * math.sqrt(context.shape[1]), torch.abs(question - context))
    else:
        raise ValueError('Unknown feature mode')
    return torch.cat((question, context, *extra), dim=-1) / 2
