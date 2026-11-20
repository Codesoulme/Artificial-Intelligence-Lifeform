"""Compare declared parameter repair and parent rollback on the integrated decoder."""

import argparse
from pathlib import Path

from audit_awake_replacement_memory import ALLOWED, apply_delta
from deepseek_awake_recall_intent import DEFAULT_DATABASE
from deepseek_continuous_memory_chat import EARLY_TIMING_THRESHOLD, load_chat
from deepseek_output_aligned_memory import MODEL, SOURCE, read, sha, torch, write
from spora.awake_recall_intent import SelfNameIntent
from spora.dna_shadow import DnaShadow, state_digest
from spora.health_gate import cases_digest, promotion_decision
from train_awake_replacement_replay import encode_cases, generate


ORDINARY = (
    ('八个橘子分成两份，每份有几个？', '四个。'),
    ('请写一句给同学的生日祝福。', '祝你生日快乐！'),
    ('请照抄这两个字，随后结束。知道', '知道'),
)
MEMORY = (
    ('沈亦平常最爱喝什么？', '香橙冰茶'),
    ('上一次记录里，沈亦平常最爱喝什么？', '甘蔗乌龙茶'),
    ('齐宁那次去了哪里？', '新荷档案馆'),
    ('你叫什么名字？', '孢子'),
)
LEVELS = (.1, .5, .8)
SEED = 21521


def monitored(chat):
    return {f'{index}.{name}': parameter for index in (6, 12)
            for name, parameter in chat.decoder.base.model.layers[index].named_parameters() if name in ALLOWED}


def snapshot(parameters):
    return {name: tensor.detach().cpu().clone() for name, tensor in parameters.items()}


@torch.no_grad()
def assign(parameters, weights):
    for name, parameter in parameters.items():
        parameter.copy_(weights[name])


def evaluate(chat, cases, reference=None):
    ordinary = [generate(chat.decoder, chat.tokenizer, prompt, 48) for _, prompt, _ in cases]
    memory = [chat.answer(question, timing_threshold=EARLY_TIMING_THRESHOLD) for question, _ in MEMORY]
    metrics = dict(facts=sum(row['action'] == 'read' and row['response'].rstrip('」。').endswith(value)
                             for row, (_, value) in zip(memory, MEMORY, strict=True)),
                   endings=sum(tokens[-1] == chat.tokenizer.eos_token_id for tokens in ordinary))
    if reference is not None:
        metrics['matched_ordinary'] = sum(a == b for a, b in zip(ordinary, reference, strict=True))
    rendered = [chat.tokenizer.decode(tokens, skip_special_tokens=True).strip() for tokens in ordinary]
    return metrics, ordinary, rendered, memory


def main(out, database, delta_path, profile):
    name_path = profile / 'self-name.pt'
    paths = (database, delta_path, SOURCE, MODEL / 'model.safetensors', name_path)
    protected = {path: sha(path) for path in paths}
    if read(delta_path.parent / 'result.json')['delta_sha256'] != protected[delta_path]:
        raise ValueError('Unexpected calibration delta digest')
    chat = load_chat(database, use_inherited_router=True, use_retrained_gate=True)
    width = chat.decoder.base.lm_head.weight.shape[1]
    chat.self_name = SelfNameIntent.load_name(name_path, width, chat.recall_fingerprint, chat.bank_specs['drink', 1])
    parameters = monitored(chat)
    parent = snapshot(parameters)
    apply_delta(chat.decoder, delta_path)
    active = snapshot(parameters)
    active_digest = state_digest(active)
    shadow = DnaShadow(parent, alpha=.5)
    shadow.update(active)
    shadow.save(out / 'ema.pt')
    shadow = DnaShadow.load(out / 'ema.pt')
    exact = DnaShadow(active)
    exact.save(out / 'exact.pt')
    exact = DnaShadow.load(out / 'exact.pt')
    cases = encode_cases(chat.tokenizer, ORDINARY)
    seal = cases_digest(dict(ordinary=ORDINARY, memory=MEMORY))
    baseline_metrics, baseline_tokens, baseline_text, baseline_read = evaluate(chat, cases)
    baseline_metrics['matched_ordinary'] = len(ORDINARY)
    print('Baseline', baseline_metrics, flush=True)
    if baseline_metrics['facts'] != len(MEMORY):
        raise ValueError('Intact candidate failed a memory anchor')
    results = []
    count = sum(tensor.numel() for tensor in active.values())
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(count, generator=generator)
    for level in LEVELS:
        mask = torch.zeros(count, dtype=torch.bool)
        mask[permutation[:round(level * count)]] = True
        masks = {}
        damaged = {}
        start = 0
        for name, tensor in active.items():
            end = start + tensor.numel()
            masks[name] = mask[start:end].reshape_as(tensor)
            damaged[name] = tensor.masked_fill(masks[name], 0.)
            start = end
        repaired_ema, repaired_counts = shadow.repair_damaged(damaged, damage_masks=masks)
        repaired_exact, _ = exact.repair_damaged(damaged, damage_masks=masks)
        variants = {}
        for label, weights in (('damaged', damaged), ('parent_rollback', parent),
                               ('ema_repair', repaired_ema), ('exact_repair', repaired_exact)):
            assign(parameters, weights)
            metrics, _, ordinary, memory = evaluate(chat, cases, baseline_tokens)
            decision = promotion_decision(baseline_metrics, metrics, expected_cases_digest=seal,
                                          candidate_cases_digest=cases_digest(dict(ordinary=ORDINARY, memory=MEMORY)))
            variants[label] = dict(metrics=metrics, decision=decision, ordinary=ordinary, memory=memory,
                                   exact_weights=state_digest(weights) == active_digest)
            print(level, label, metrics, decision, flush=True)
        if not variants['exact_repair']['exact_weights'] or not variants['exact_repair']['decision']['promote']:
            raise AssertionError('Exact local recovery diverged from the active anchor')
        results.append(dict(damage_fraction=level, damaged_parameters=sum(repaired_counts.values()), variants=variants))
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected model or memory file changed')
    write(out / 'result.json', dict(seal=seal, monitored_parameters=count, baseline=baseline_metrics,
                                   baseline_ordinary=baseline_text, baseline_memory=baseline_read,
                                   sources={str(path): digest for path, digest in protected.items()},
                                   shadow_bytes={name: (out / name).stat().st_size for name in ('ema.pt', 'exact.pt')},
                                   results=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--delta', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    main(args.output, args.database, args.delta, args.profile)
