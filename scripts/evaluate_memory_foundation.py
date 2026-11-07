"""P25 structural witnesses, independent probes, and exact legacy checks.

No language training or candidate promotion. Probe individuals are synthetic
batched matrices for an information-preservation test, not saved life histories.
"""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F

from spora.conversation import ConversationBridge, ConversationConfig, generate, restore_conversation
from spora.growing import GrowingState, grow, load, new_state
from spora.language_training import frozen_digest
from spora.sleep_memory import SealedPage
from spora.temporal import TemporalConfig, TemporalSPORA
from spora.typed_memory import TypedMemoryReader, frozen_value_read

ROOT = Path(__file__).resolve().parents[1]
BASE = '1985f436b63f0afe58a4c3d3be7f97ec0b592581'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fixture():
    core = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2,
        memory_width=64, expert_rank=4, memory_layers=(0, 1),
        history_writer='rls', current_address='orthogonal'))
    state = new_state(core)
    state.cursor = 1
    config = ConversationConfig(vocabulary=64, width=32, depth=2, heads=4,
        hidden=48, context=32, memory_layers=(0, 1))
    return core, state, config


def structural_witness():
    torch.manual_seed(25)
    core, state, config = fixture()
    state.current[0] = torch.randn_like(state.current[0])
    state.pages[0][0][0] = torch.randn_like(state.pages[0][0][0])
    swapped = copy.deepcopy(state)
    swapped.current[0], swapped.pages[0][0][0] = swapped.pages[0][0][0], swapped.current[0]
    x = torch.randn(2, 5, 32)
    results = {}
    for kind in ('legacy', 'typed_v1'):
        model = ConversationBridge(core, replace(config, memory_read=kind))
        with torch.no_grad():
            baseline = model.read_memory(x, state, 0)
            role = model.read_memory(x, swapped, 0)
            extra = model.read_memory(x, grow(core, state, 3), 0)
        results[kind] = dict(role_swap_max_abs=float((baseline-role).abs().max()),
            unused_growth_max_abs=float((baseline-extra).abs().max()))
    return results


def probe_data(seed, pairs=256, width=32):
    generator = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(pairs, 1, width, generator=generator), dim=-1)
    # First value coordinate is the synthetic fact to recall. Other coordinates
    # are independent distractors. A pair swaps current/history, keeps query fixed.
    a = .15*torch.randn(pairs, width, generator=generator)
    b = .15*torch.randn(pairs, width, generator=generator)
    a[:, 0] = 1
    b[:, 0] = -1
    current = torch.cat((a, b))[:, :, None] * torch.cat((q, q))
    history = torch.cat((b, a))[:, :, None] * torch.cat((q, q))
    labels = torch.cat((torch.ones(pairs), torch.zeros(pairs))).long()
    return torch.cat((q, q)), GrowingState([current], [[[history]]], 1), labels


def probe_features(q, state, kind):
    if kind == 'typed_v1':
        reader = TypedMemoryReader(q.shape[-1], sites=1)
        return reader(q, state, 0, 64)[:, 0].detach()
    values = torch.stack((frozen_value_read(q, state.current[0]),
                          frozen_value_read(q, state.pages[0][0][0])), 2)
    weights = (values*q.unsqueeze(2)).sum(-1).softmax(-1)
    mixed = (values*weights.unsqueeze(-1)).sum(2)[:, 0]
    # Equal downstream parameter budget; duplicating the ambiguous read does not
    # recover the lost role. No source label is appended to either input.
    return torch.cat((mixed, mixed), -1).detach()


def role_probe(seed):
    train_q, train_state, train_y = probe_data(seed)
    test_q, test_state, test_y = probe_data(seed+100000)
    result = {'seed': seed, 'train_pairs': 256, 'held_pairs': 256, 'steps': 200}
    for kind in ('legacy', 'typed_v1'):
        x = probe_features(train_q, train_state, kind)
        test_x = probe_features(test_q, test_state, kind)
        torch.manual_seed(seed)
        head = torch.nn.Linear(x.shape[-1], 2)
        opt = torch.optim.Adam(head.parameters(), lr=.02)
        for _ in range(200):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x), train_y)
            loss.backward()
            opt.step()
        with torch.no_grad():
            result[kind] = dict(accuracy=float((head(test_x).argmax(-1)==test_y).float().mean()),
                paired_feature_max_abs=float((test_x[:256]-test_x[256:]).abs().max()),
                head_parameters=sum(p.numel() for p in head.parameters()))
    return result


def factor_measurement():
    torch.manual_seed(2511)
    width, rank = 90, 16
    left = torch.randn(width, rank)*.1
    right = torch.randn(rank, width)*.1
    page = SealedPage([[left, right, torch.zeros(width*(width+1)//2)]], rank)
    results = []
    for length in (1, 128):
        q = F.normalize(torch.randn(1, length, width), dim=-1)
        def old():
            return frozen_value_read(q, page.materialize(0)[0])
        def new():
            return frozen_value_read(q, (left, right))
        with torch.no_grad():
            error = float((old()-new()).abs().max())
            for _ in range(20): old(); new()
            samples = {'dense': [], 'factor': []}
            for trial in range(9):
                arms = [('dense', old), ('factor', new)]
                if trial % 2: arms.reverse()
                for name, call in arms:
                    start = time.perf_counter()
                    for _ in range(100): call()
                    samples[name].append((time.perf_counter()-start)/100)
        results.append(dict(tokens=length, width=width, rank=rank, max_abs_error=error,
            median_seconds={k:statistics.median(v) for k,v in samples.items()}, samples_seconds=samples))
    return results


def real_checkpoint_compatibility():
    parent = ROOT/'checkpoints/individual-prechinese-v3.pt'
    checkpoint = ROOT/'runs/p23f-small/step002000.pt'
    before_files = {str(p.relative_to(ROOT)):sha(p) for p in (parent, checkpoint)}
    core, state, _ = load(parent)
    before = frozen_digest(core, state)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    new = restore_conversation(core, payload, before_files[str(parent.relative_to(ROOT))], payload['tokenizer_sha256'])
    # Load exactly the reviewed pre-change module, with its relative imports.
    source = subprocess.check_output(['git', 'show', BASE+':spora/conversation.py'], cwd=ROOT, text=True, encoding='utf-8')
    baseline = types.ModuleType('spora._p25_legacy_reference')
    sys.modules[baseline.__name__] = baseline
    exec(compile(source, BASE+':spora/conversation.py', 'exec'), baseline.__dict__)
    old = baseline.restore_conversation(copy.deepcopy(core), payload,
        before_files[str(parent.relative_to(ROOT))], payload['tokenizer_sha256'])
    del payload
    tokens = torch.tensor([[2, 3, 319, 700, 1000, 8, 90, 80]])
    with torch.no_grad():
        expected = old(tokens, state)
        actual = new(tokens, state)
        old_gen = baseline.generate(old, state, tokens[0].tolist(), eos=1, budget=12)
        new_gen = generate(new, state, tokens[0].tolist(), eos=1, budget=12)
    files_same = all(sha(ROOT/name)==digest for name,digest in before_files.items())
    return dict(baseline_commit=BASE, checkpoint_sha256=before_files[str(checkpoint.relative_to(ROOT))],
        logits_bitwise_equal=torch.equal(expected, actual), generated_tokens_equal=old_gen==new_gen,
        original_tensors_exact=before==frozen_digest(core, state), original_files_unchanged=files_same,
        scope='Actual P23f-small2000 checkpoint; numeric token prefix for regression, not language-quality grading.')


def main():
    torch.set_num_threads(1)
    out = ROOT/'runs/p25-foundation/experiment.json'
    if out.exists(): raise FileExistsError(out)
    result = dict(structural_witness=structural_witness(),
        role_probes=[role_probe(seed) for seed in (7, 11, 23)],
        factor_read=factor_measurement(), legacy_checkpoint=real_checkpoint_compatibility(),
        scope='Representation and lifecycle foundations only. No claim of Chinese improvement, semantic alignment, calibrated evidence, or autonomous general reasoning.')
    w = result['structural_witness']
    result['passed'] = (w['legacy']['role_swap_max_abs']==0
        and w['typed_v1']['role_swap_max_abs']>1e-5
        and w['typed_v1']['unused_growth_max_abs']==0
        and all(p['legacy']['accuracy']==.5 and p['typed_v1']['accuracy']>=.98 for p in result['role_probes'])
        and all(r['max_abs_error']<1e-5 for r in result['factor_read'])
        and all(result['legacy_checkpoint'][k] for k in ('logits_bitwise_equal','generated_tokens_equal','original_tensors_exact','original_files_unchanged')))
    paths = ['spora/conversation.py','spora/conversation_growth.py','spora/typed_memory.py',
        'spora/conversation_state.py','tests/test_memory_foundation.py','scripts/evaluate_memory_foundation.py']
    result['sources'] = {p:sha(ROOT/p) for p in paths}
    out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
    if not result['passed']: raise SystemExit('Foundation experiment failed')


if __name__ == '__main__':
    main()
