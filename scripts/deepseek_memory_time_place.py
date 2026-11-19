"""Test contrastive temporal routing and inspect a failed memory token."""

import argparse
import json
import time
from pathlib import Path

from deepseek_aligned_memory_dialogue import RANK, matrix_bias
from deepseek_memory_attribute_span import KEYS, Route, batch, fit, route_action
from deepseek_memory_dialogue_context import OUT as CONTEXT_RUN, features_for, route
from deepseek_memory_field_transfer import OUT as TRANSFER_RUN
from deepseek_memory_turn_priority import OUT as PRIORITY_RUN, combine, gate_probabilities
from deepseek_output_aligned_memory import (
    MODEL, ROOT, SEED, SLOTS, SOURCE, encode, fingerprint, fresh, load_integrated, read, sha, torch, write, write_bank,
)

OUT = ROOT / 'runs/deepseek-memory-time-place'
TRAIN_PAIRS = (
    ('这条访谈写到他到会场报到。我要核对他', '报到的年月日。', '报到的场馆名。', '请把这一栏读出来。'),
    ('那份登记记录了她去接待站的经过。我得确认', '她几时到站。', '她到了哪处接待站。', '这上面填的是什么？'),
    ('他们坐车去参加活动。我在找', '车几点到。', '车开到了哪儿。', '那项能念一下吗？'),
    ('我在核查那回参观的过程，想知道', '参观安排在哪个星期。', '参观的是哪家展馆。', '你把查出来的结果说说。'),
    ('他那回从驻地出门去办事。我要的是', '动身的钟点。', '办事的地址。', '有这项资料吗？'),
    ('看看她最后抵达的记录，主要核对', '到达时刻。', '终点地名。', '麻烦说下记的哪项。'),
)
HELD_PAIRS = (
    ('物流笔记写了师傅下午去卸货，我要确认', '真正到货是周几。', '卸货的是哪个站点。', '能帮我从原文找一下吗？'),
    ('现场纪要说调查组到镇上碰头，我要查', '碰头那天的日子。', '碰头的具体去处。', '请念出相应条目。'),
    ('她坐渡船抵达后接受采访，我现在问的是', '靠岸属于哪一日。', '靠的是哪个码头。', '这部分有写明白吗？'),
    ('他在旧行程表上标出了首站，需要核实', '抵达首站的时间。', '首站究竟在哪儿。', '这一格具体填的啥？'),
    ('那趟考察队赶去开会，我关心材料里', '会议是几月开的。', '会在哪个地方开。', '这个细节你还能读到吗？'),
    ('旅客在途中登记了一次到站信息。我要查', '到站时是上午还是下午。', '到的是哪座车站。', '能说一下原记录吗？'),
)


def paired_rows(prefix, pairs):
    return [dict(key=f'{prefix}-{i}-{field}', history=stem + clause, question=question, field=field, target_turn=0)
            for i, (stem, when, where, question) in enumerate(pairs)
            for field, clause in (('time', when), ('place', where))]


@torch.inference_mode()
def token_diagnostic(decoder, tokenizer, head, out):
    plan = read(TRANSFER_RUN / 'plan.json')
    transfer = read(TRANSFER_RUN / 'result.json')
    row = next(r for r in transfer['rows'] if r['key'] == 'held-3' and r['trial'] == 'known_field'
               and r['mode'] == 'field' and r['version'] == 1)
    fixture = next(r for r in plan['rows'] if r['key'] == row['key'])
    inherits = next(r for r in read(TRANSFER_RUN / 'routes.json') if r['key'] == row['key'])['inherits']
    prefix = features_for(decoder, tokenizer, fixture, inherits)[3] + row['tokens'][:row['start']]
    logits = decoder(input_ids=torch.tensor([prefix], device='cuda'), use_cache=False).logits[0, -1].float()
    target = encode(tokenizer, row['expected'])
    first, actual = target[0], row['tokens'][row['start']]
    organism, _ = fresh()
    site = organism.memories[0]
    address = site.current_basis[row['entity']:row['entity'] + 1]
    basis = torch.linalg.qr(torch.randn(head.weight.shape[1], RANK, generator=torch.Generator().manual_seed(SEED))).Q.to('cuda')
    banks = torch.load(TRANSFER_RUN / 'banks.pt', weights_only=True)
    bank = banks[row['relation'], row['version']]
    directions = torch.zeros_like(bank[:, :, 0])
    ids = [*target, tokenizer.eos_token_id]
    directions[:len(ids), :RANK] = torch.nn.functional.normalize(head.weight[ids].float() @ basis, dim=-1).cpu()
    isolated = write_bank(site, torch.zeros_like(bank), address, directions)
    observations = []
    for mode, matrices in (('shared_bank', bank), ('single_address', isolated)):
        bias = matrix_bias(head, site, matrices, address, basis)[0]
        for gain in (0., 1., 2., 3., 4.):
            scores = logits + gain * bias
            winner = int(scores.argmax())
            observations.append(dict(mode=mode, gain=gain, winner=winner, winner_text=tokenizer.decode([winner]),
                                     target_rank=1 + int((scores > scores[first]).sum()),
                                     target_margin=float(scores[first] - scores[winner]),
                                     raw_target=float(logits[first]), raw_actual=float(logits[actual]),
                                     memory_target=float(bias[first]), memory_actual=float(bias[actual])))
            if mode == 'shared_bank' and gain == 2. and winner != actual:
                raise ValueError('Failed to reproduce original first token')
    write(out / 'token-diagnostic.json', dict(case=row, target=first, target_text=tokenizer.decode([first]),
          actual=actual, actual_text=tokenizer.decode([actual]), observations=observations))


def main(out):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    torch.set_num_threads(2)
    priority_plan = read(PRIORITY_RUN / 'plan.json')
    setup = read(PRIORITY_RUN / 'routes.json')
    source_paths = (PRIORITY_RUN / 'plan.json', PRIORITY_RUN / 'routes.json', PRIORITY_RUN / 'routing.pt',
                    CONTEXT_RUN / 'router.pt', TRANSFER_RUN / 'plan.json', TRANSFER_RUN / 'routes.json',
                    TRANSFER_RUN / 'banks.pt', TRANSFER_RUN / 'result.json')
    for path in source_paths[:-1]:
        if sha(path) != read(path.parent / 'result.json')['artifact_sha256'][path.name]:
            raise ValueError(f'Source artifact changed: {path}')
    inputs = (Path(__file__), SOURCE, *source_paths,
              *(MODEL / name for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')))
    hashes = fingerprint(inputs)
    train = priority_plan['train'] + paired_rows('train-pair', TRAIN_PAIRS)
    suites = {'held_pairs': paired_rows('held-pair', HELD_PAIRS), 'transfer': read(TRANSFER_RUN / 'plan.json')['rows'],
              'prior': priority_plan['held']}
    if {r['question'] for r in train} & {r['question'] for rows in suites.values() for r in rows}:
        raise ValueError('Train and held questions overlap')
    out.mkdir()
    write(out / 'plan.json', dict(train=train, suites=suites, protected=hashes))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
              dtype=torch.float16, attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, state = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    del parent, state
    decoder.eval().requires_grad_(False)
    width = base.lm_head.weight.shape[1]
    original, direct = (Route(width, localized=True) for _ in range(2))
    original.load_state_dict(torch.load(CONTEXT_RUN / 'router.pt', weights_only=True))
    priority = torch.load(PRIORITY_RUN / 'routing.pt', weights_only=True)
    direct.load_state_dict(priority['direct'])
    gate = torch.nn.Linear(width, 1).eval()
    gate.load_state_dict(priority['inheritance'])
    full = {r['key']: features_for(decoder, tokenizer, r, True) for r in train}
    training = batch([dict(question=r['key']) for r in train], full)
    candidate = fit(training, torch.tensor([KEYS.index(r['field']) for r in train]), localized=True)
    torch.save(candidate.state_dict(), out / 'router.pt')
    candidate.load_state_dict(torch.load(out / 'router.pt', weights_only=True))
    cutoff = setup['cutoffs']['priority']
    scored = {}
    for name, rows in suites.items():
        full = {r['key']: features_for(decoder, tokenizer, r, True) for r in rows}
        current = {r['key']: features_for(decoder, tokenizer, r, False) for r in rows}
        single = route(direct.eval(), rows, current)
        probabilities = gate_probabilities(gate, rows, current)
        scored[name] = {}
        for mode, router in (('original', original.eval()), ('augmented', candidate)):
            combined = combine(route(router, rows, full), single, probabilities, setup['gate_cutoff'])
            scored[name][mode] = [dict(r, action=route_action(r, cutoff)) for r in combined]
        print('Evaluated', name, flush=True)
    token_diagnostic(decoder, tokenizer, base.lm_head, out)
    if fingerprint(inputs) != hashes:
        raise ValueError('Experiment inputs changed')
    write(out / 'result.json', dict(routes=scored, cutoff=cutoff, elapsed_seconds=time.perf_counter() - started,
          peak_allocated_bytes=torch.cuda.max_memory_allocated(), artifact_sha256={name: sha(out / name) for name in
          ('plan.json', 'router.pt', 'token-diagnostic.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    main(parser.parse_args().out)
