"""Named tensor state for transactional storage; no event/answer records."""
import torch
from .growing import GrowingState
from .temporal import TemporalSPORA, TemporalConfig


def flatten(model, state):
    if model.config.temporal_kind != 'split' or len(state.current) != len(model.memories):
        raise ValueError('Split singleton state required')
    if type(state.cursor) is not int or not 1 <= len(state.pages) <= 16 or not 0 <= state.cursor <= len(state.pages)*model.config.time_count:
        raise ValueError('Invalid state extent')
    tensors = {'model/'+name: value for name, value in model.state_dict(keep_vars=True).items()}
    for site, tensor in enumerate(state.current):
        if tensor.shape != (1, *model.memories[site].initial[0].shape):
            raise ValueError('Invalid current shape')
        tensors[f'current/{site}'] = tensor
    pages = []
    for page_index, page in enumerate(state.pages):
        if not isinstance(page, list):
            from .sleep_memory import SealedPage
            if not isinstance(page, SealedPage):
                raise ValueError('Unknown page type')
            page.validate(model)
            if (page_index+1)*model.config.time_count > state.cursor:
                raise ValueError('Cannot seal unfinished page')
            pages.append({'kind': 'sealed', 'rank': page.rank})
            sites = page.factors
        else:
            pages.append({'kind': 'active'})
            sites = page
            if len(sites) != len(model.memories):
                raise ValueError('Incorrect memory layer count')
            for site, params in enumerate(sites):
                if len(params) != len(model.memories[site].initial)-1:
                    raise ValueError('Incorrect page tensor count')
                for tensor, initial in zip(params, model.memories[site].initial[1:]):
                    if tensor.shape != (1, *initial.shape):
                        raise ValueError('Invalid active page shape')
        for site, params in enumerate(sites):
            for index, tensor in enumerate(params):
                tensors[f'page/{page_index}/{site}/{index}'] = tensor
    config = model.config_dict()
    config['memory_layers'] = list(config['memory_layers'])
    schema = {'format': 'spora-tensors-v1', 'config': config,
              'cursor': state.cursor, 'pages': pages}
    return schema, tensors


def restore(schema, tensors, device='cpu'):
    if schema.get('format') != 'spora-tensors-v1':
        raise ValueError('Unsupported tensor state')
    model = TemporalSPORA(TemporalConfig(**schema['config'])).to(device)
    model.load_state_dict({k[6:]: v for k, v in tensors.items() if k.startswith('model/')})
    model.eval()
    current = [tensors[f'current/{site}'].to(device) for site in range(len(model.memories))]
    pages = []
    for page_index, description in enumerate(schema['pages']):
        count = 3 if description['kind'] == 'sealed' else len(model.memories[0].initial)-1
        sites = [[tensors[f'page/{page_index}/{site}/{index}'].to(device)
                  for index in range(count)] for site in range(len(model.memories))]
        if description['kind'] == 'sealed':
            from .sleep_memory import SealedPage
            pages.append(SealedPage(sites, description['rank']))
        elif description['kind'] == 'active':
            pages.append(sites)
        else:
            raise ValueError('Unknown stored page kind')
    state = GrowingState(current, pages, schema['cursor'])
    check, expected = flatten(model, state)
    if set(tensors) != set(expected):
        raise ValueError('Missing or unexpected tensors')
    if type(state.cursor) is not int or not 1 <= len(pages) <= 16 or not 0 <= state.cursor <= len(pages)*model.config.time_count:
        raise ValueError('Invalid state extent')
    for site, tensor in enumerate(current):
        if tensor.shape != (1, *model.memories[site].initial[0].shape):
            raise ValueError('Invalid current shape')
    for p, page in enumerate(pages):
        if isinstance(page, list):
            for site, params in enumerate(page):
                for tensor, initial in zip(params, model.memories[site].initial[1:]):
                    if tensor.shape != (1, *initial.shape):
                        raise ValueError('Invalid active page shape')
        else:
            page.validate(model)
            if (p+1)*model.config.time_count > state.cursor:
                raise ValueError('Cannot seal unfinished page')
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors.values()):
        raise ValueError('Nonfinite tensor state')
    return model, state
