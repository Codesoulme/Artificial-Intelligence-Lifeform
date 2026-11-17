"""Temporary ablations for isolated memory-reader research only."""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

PATHS = frozenset(('native', 'low_rank', 'direct', 'final'))


@contextmanager
def ablate_memory_paths(reader, low_rank, direct, final, disabled):
    """Bypass selected modules and restore all forward methods on exit.

    The bypass of a residual adapter invokes its own base operation; the
    native reader is an additive hidden-state operation and becomes identity.
    This changes no tensor or saved checkpoint and is not a runtime switch.
    """
    disabled = frozenset(disabled)
    if not disabled <= PATHS:
        raise ValueError('Unknown memory path')
    lists = {'low_rank': tuple(low_rank), 'direct': tuple(direct), 'final': tuple(final)}
    modules = [module for group in lists.values() for module in group]
    if len({id(module) for module in modules}) != len(modules):
        raise ValueError('Adapters must be distinct')
    if any(not callable(getattr(module, 'base', None)) for module in modules):
        raise ValueError('Adapter missing its base operation')
    with ExitStack() as stack:
        if 'native' in disabled:
            stack.enter_context(patch.object(reader, 'forward', side_effect=lambda hidden, *args, **kwargs: hidden))
        for name, group in lists.items():
            if name in disabled:
                for module in group:
                    stack.enter_context(patch.object(module, 'forward',
                                                     side_effect=lambda x, module=module: module.base(x)))
        yield
