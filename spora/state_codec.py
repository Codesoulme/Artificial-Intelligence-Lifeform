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
    if hasattr(model, 'chinese_query'):
        model.chinese_query.config.validate()
        schema['chinese_query'] = model.chinese_query.config_dict()
    if hasattr(model, 'named_address_binding'):
        from .named_address import NamedAddressBinding
        if not isinstance(model.named_address_binding, NamedAddressBinding):
            raise ValueError('Invalid named address binding')
        model.named_address_binding.validate(model)
        schema['named_address_binding'] = model.named_address_binding.config_dict()
    if hasattr(model, 'identity_fingerprints'):
        from .identity_fingerprints import IdentityFingerprints
        if not isinstance(model.identity_fingerprints, IdentityFingerprints):
            raise ValueError('Invalid identity fingerprint module')
        model.identity_fingerprints.validate(model)
        schema['identity_fingerprints'] = model.identity_fingerprints.config_dict()
    if hasattr(model, 'historical_resolutions'):
        from .historical_resolutions import HistoricalResolutions
        if not isinstance(model.historical_resolutions, HistoricalResolutions):
            raise ValueError('Invalid historical resolution module')
        model.historical_resolutions.validate(model, state)
        schema['historical_resolutions'] = model.historical_resolutions.config_dict()
    if hasattr(model, 'awake_core'):
        model.awake_core.validate()
        schema['awake_core'] = model.awake_core.config_dict()
    if hasattr(model, 'semantic_memory'):
        module = model.semantic_memory
        if not -1 <= int(module.trained_cursor) <= state.cursor or not 0 <= int(module.evidence_count) <= 64:
            raise ValueError('Invalid semantic revision')
        schema['semantic_memory'] = module.config_dict()
    if hasattr(model, 'skill_policy'):
        if int(model.skill_policy.feedback_steps) < 0:
            raise ValueError('Invalid skill training counter')
        schema['skill_policy'] = model.skill_policy.config_dict()
    if hasattr(model, 'time_world') != (state.runtime is not None):
        raise ValueError('Time module and runtime state must travel together')
    if state.runtime is not None:
        state.runtime.validate()
        schema['time_world'] = model.time_world.config_dict()
        schema['runtime'] = state.runtime.metadata()
        tensors.update({'runtime/'+k: v for k, v in state.runtime.tensors.items()})
    if hasattr(model,'write_control') != (state.internal is not None):
        raise ValueError('Write controller and internal state must travel together')
    if state.internal is not None:
        state.internal.validate(model,state)
        schema['write_control']=model.write_control.config_dict()
        schema['internal']=state.internal.metadata()
        tensors['internal/evidence']=state.internal.evidence
        for p,tensor in enumerate(state.internal.provenance):
            tensors[f'internal/provenance/{p}']=tensor
        if state.internal.router_factors is not None:
            for site,pages in enumerate(state.internal.router_factors):
                for page,pair in enumerate(pages):
                    for i,tensor in enumerate(pair):
                        tensors[f'internal/router/{site}/{page}/{i}']=tensor
    return schema, tensors


def restore(schema, tensors, device='cpu'):
    if schema.get('format') != 'spora-tensors-v1':
        raise ValueError('Unsupported tensor state')
    model = TemporalSPORA(TemporalConfig(**schema['config'])).to(device)
    if 'chinese_query' in schema:
        from .chinese_query import ChineseQuery, ChineseQueryConfig
        model.chinese_query = ChineseQuery(ChineseQueryConfig(**schema['chinese_query'])).to(device)
    if 'named_address_binding' in schema:
        from .named_address import NamedAddressBinding
        model.named_address_binding = NamedAddressBinding(**schema['named_address_binding']).to(device)
    if 'identity_fingerprints' in schema:
        from .identity_fingerprints import IdentityFingerprints
        model.identity_fingerprints = IdentityFingerprints(**schema['identity_fingerprints']).to(device)
    if 'historical_resolutions' in schema:
        from .historical_resolutions import HistoricalResolutions
        model.historical_resolutions = HistoricalResolutions(**schema['historical_resolutions']).to(device)
    if 'awake_core' in schema:
        from .awake import AwakeCore, AwakeConfig
        model.awake_core = AwakeCore(AwakeConfig(**schema['awake_core'])).to(device)
    if 'semantic_memory' in schema:
        from .semantic_memory import SemanticMemory
        model.semantic_memory = SemanticMemory(**schema['semantic_memory']).to(device)
    if 'skill_policy' in schema:
        from .skills import FeedbackPolicy
        model.skill_policy = FeedbackPolicy(**schema['skill_policy']).to(device)
    if ('time_world' in schema) != ('runtime' in schema):
        raise ValueError('Incomplete time extension')
    if 'time_world' in schema:
        from .world_time import TimeWorld, WorldTimeConfig
        model.time_world = TimeWorld(WorldTimeConfig(**schema['time_world'])).to(device)
    if ('write_control' in schema) != ('internal' in schema):
        raise ValueError('Incomplete internal extension')
    if 'write_control' in schema:
        from .internal_memory import NoveltyWriteControl, WriteControl
        config = dict(schema['write_control'])
        controller = NoveltyWriteControl if config.pop('kind', None) == 'novelty' else WriteControl
        model.write_control=controller(**config).to(device)
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
    if 'runtime' in schema:
        from .online_time import OnlineState
        state.runtime = OnlineState(**schema['runtime'], tensors={k[8:]: v.to(device) for k, v in tensors.items() if k.startswith('runtime/')})
    if 'internal' in schema:
        from .internal_memory import InternalState
        factors=None
        if schema['internal']['router_cursor'] >= 0:
            factors=[[[tensors[f'internal/router/{site}/{page}/{i}'].to(device) for i in range(2)]
                       for page in range(len(pages)+1)] for site in range(len(model.memories))]
        state.internal=InternalState(evidence=tensors['internal/evidence'].to(device),
            provenance=[tensors[f'internal/provenance/{p}'].to(device) for p in range(len(pages))],
            router_factors=factors,**dict({'encoding':1},**schema['internal']))
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
