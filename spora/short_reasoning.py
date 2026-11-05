"""Bounded, supervised evidence traces and an optional recurrent language block.

The task generator/checker lives in the experiment script. Inference receives
only input bytes and existing neural state, never a solved event table.
"""
import torch
from torch.nn import functional as F
from .model import CausalBlock
from .language_bridge import LanguageBridge, BOS, EOS, PAD, VOCAB
from .language_training import load_interface, interface_state


class RecurrentLanguageBridge(LanguageBridge):
    def __init__(self, organism, passes=2, source=None, parent_sha256=None):
        if type(passes) is not int or not 1 <= passes <= 3:
            raise ValueError('Bounded recurrent depth required')
        super().__init__(organism)
        # Explicitly adopt an ordinary interface before attaching new weights.
        if source is not None:
            load_interface(self, source, parent_sha256)
        self.passes = passes
        self.reasoner = CausalBlock(organism.config)
        with torch.no_grad():
            for layer in (self.reasoner.out, self.reasoner.ff[-1]):
                layer.weight.zero_()
                layer.bias.zero_()

    def contextualize(self, x):
        for _ in range(self.passes):
            x = self.reasoner(x)
        return x


def response_loss(logits, tokens, starts):
    """Score only supplied response bytes/EOS, not input or padding."""
    if tokens.ndim != 2 or starts.shape != (len(tokens),) or starts.dtype != torch.long:
        raise ValueError('Response start per sequence required')
    if bool(((starts < 1) | (starts >= tokens.shape[1])).any()):
        raise ValueError('Response start outside sequence')
    targets = tokens[:, 1:].clone()
    positions = torch.arange(1, tokens.shape[1], device=tokens.device)[None]
    targets[(positions < starts[:, None]) | (targets == PAD)] = -100
    if bool((targets != -100).sum(1).eq(0).any()):
        raise ValueError('Each response needs at least one target')
    return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), targets.reshape(-1), ignore_index=-100)


@torch.no_grad()
def generate_equal_prefixes(model, state, prefixes, budget=48):
    """Efficient raw generation; equal lengths avoid padding in attention."""
    if not prefixes or len({len(p) for p in prefixes}) != 1:
        raise ValueError('Nonempty equally sized prefixes required')
    if type(budget) is not int or not 1 <= budget <= 128 or len(prefixes[0])+budget > model.organism.config.max_length:
        raise ValueError('Invalid generation budget')
    tokens = torch.tensor(prefixes, dtype=torch.long, device=next(model.parameters()).device)
    finished = torch.zeros(len(prefixes), dtype=torch.bool, device=tokens.device)
    outputs = [[] for _ in prefixes]
    for _ in range(budget):
        logits = model(tokens, state)[:, -1].clone()
        if not bool(torch.isfinite(logits).all()):raise ValueError('Nonfinite output')
        logits[:, [BOS, PAD]] = -torch.inf
        selected = logits.argmax(-1)
        for i, token in enumerate(selected.tolist()):
            if not bool(finished[i]) and token != EOS:outputs[i].append(token)
        finished |= selected == EOS
        tokens = torch.cat([tokens, selected[:, None]], 1)
        if bool(finished.all()):break
    results = []
    for output, ended in zip(outputs, finished.tolist()):
        raw = bytes(output)
        try:text = raw.decode('utf-8'); valid = True
        except UnicodeDecodeError:text = raw.decode('utf-8', errors='replace'); valid = False
        results.append({'text': text, 'bytes_hex': raw.hex(), 'utf8_valid': valid, 'ended_with_eos': ended})
    return results


def restore_research_interface(organism, payload, parent_sha256, device='cpu'):
    if payload.get('format') != 'spora-short-reasoning-v1' or payload.get('arm') not in ('answer','trace','recurrent_trace'):
        raise ValueError('Invalid research interface format')
    if payload['arm'] == 'recurrent_trace':
        model = RecurrentLanguageBridge(organism, payload['passes'])
    else:
        if payload['passes'] != 0:raise ValueError('Unexpected recurrence')
        model = LanguageBridge(organism)
    # Existing strict loader validates lineage, complete tensor names, dtypes,
    # shape and finite values for the explicitly constructed architecture.
    load_interface(model, {'format':'spora-byte-interface-v1',
        'organism_sha256':payload['organism_sha256'],'state':payload['state']}, parent_sha256)
    return model.to(device).eval()
