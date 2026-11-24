"""Compare a separate memory-access decision on matched Chinese requests and mentions."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_native_version_families import load_schedules
from scripts.audit_spora_version_clarification import (HELD as PRIOR_HELD, SEED, action_head, examples,
                                                      run_dialogues, train_head, summarize)
from scripts.audit_spora_version_roles import (CAL_NAMES, TRAIN_NAMES, base, clause_features, flat_features,
                                              role_predictions)
from scripts.audit_spora_version_timeline import SUBJECTS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_clarification import Intent, VersionAccessHead, scope_version_intent
from spora.research_version_roles import VersionRoleHead
from spora.research_versioned_memory_pointer import VersionReplyActions


TRAIN = (
    ('{q}登记的是哪项？', '这次请实际查{v}。{p}', '我想查一版，先跟我确认该看哪份。{p}',
     '这件事明天再查。{p}', '我想判断这句有没有歧义。{p}'),
    ('{q}这栏写了什么？', '拿{v}来回答。{p}', '我想知道里面的值，但没说定用哪一版。{p}',
     '{p}先把这问题存着，有空我再问。', '{p}这句给小学生看，读得懂吗？'),
    ('{q}记录里填的是啥？', '按{v}看一眼。{p}', '这个要查，我没想好从哪次记录开始。{p}',
     '{p}不急着知道，等我招呼再看。', '给人举个提问的例子。{p}你觉得这个问法读着顺吗？'),
    ('{q}写的是哪一项？', '{p}请给我查{v}的结果。', '{p}但我说的是哪个时候还不清楚，你先问我。',
     '我先攒个待查清单。{p}这条放在明天。', '{p}帮我看看这条适合做题目吗。'),
    ('{q}记的是哪一条？', '告诉我{v}那份。{p}', '这里可能指两次记录。{p}先问我一下是哪次再答。',
     '我们暂时把查询留着。{p}', '试读下面这个问题，看看搭配奇不奇怪。{p}'),
    ('{q}还记得是哪项吗？', '我需要{v}的值。{p}', '我决定核对，先跟我定要用的版次。{p}',
     '等我改天回来再接着查。{p}', '读一读问句，语气是不是有点生硬？{p}'),
)
CALIBRATION = (
    ('{q}记录的是哪项？', '只给{v}的那一项。{p}', '我得查一下，但我自己没说清是哪一版。{p}',
     '我现在腾不出手，暂留这条。{p}', '下面的问句有没有冒犯到人？{p}'),
    ('{q}当时填的是哪条？', '帮忙核一下{v}。{p}', '想先确认该用哪一份，然后再查。{p}',
     '先记个待办，我之后会接着问。{p}', '这一句放进对话合适吗？{p}'),
)
HELD = (
    ('{q}记的是什么？', '我现在要结果，照{v}来。{p}', '{p}这次想查，但我分不清要哪一次，先和我对一下。',
     '等我忙完这阵再翻出来。{p}', '看看下面问句写得够不够清楚。{p}'),
    ('{q}录下的内容是哪项？', '{p}给我说一下{v}的。', '有两份我都拿不准，我要查其中一份。{p}',
     '{p}问题先放这里，等会儿有时间我再回来问。', '{p}这一句念起来拗口吗？'),
    ('{q}写的是什么？', '问题请按{v}答。{p}', '我真想核实这条，先定一下到底是哪次。{p}',
     '这是后面要处理的题，咱们先绕过。{p}', '帮我用更口语的方式表达下面的问法。{p}'),
)
VERSIONS = ('当前版', '首版', '上一版')
THRESHOLDS = (0., .35, .50, .65, .80, .90, .97, .99, 1.01)


def matched_examples(families, names):
    rows = []
    for family, (payload, *frames) in enumerate(families):
        for subject, (name, field) in enumerate(names):
            version = (family + subject) % len(VERSIONS)
            question = payload.format(q=f'{name}的{field}')
            for intent, frame in zip(Intent, frames, strict=True):
                rows.append({'question': frame.format(p=question, v=VERSIONS[version]), 'subject': subject,
                             'action': intent, 'version': version if intent == Intent.READ else None,
                             'family': family, 'payload': question})
    return rows


def can_read(action):
    return action in (Intent.READ, Intent.ASK)


def fit_access(head, features, rows):
    labels = torch.tensor([can_read(row['action']) for row in rows], dtype=features.dtype)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    rng = torch.Generator().manual_seed(SEED)
    # Sample whole matched groups, so the same query payload occurs on both sides of each batch.
    grouped = torch.arange(len(rows)).view(-1, len(Intent))
    for _ in range(200):
        indices = grouped[torch.randperm(len(grouped), generator=rng)[:8]].flatten()
        loss = F.binary_cross_entropy_with_logits(head(features[indices]).flatten(), labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print('Access head loss', float(loss.detach()), flush=True)


def calibrate(probabilities, rows):
    def score(cutoff):
        guesses = [(p >= cutoff, can_read(row['action'])) for p, row in zip(probabilities, rows, strict=True)]
        return (-sum(yes and not gold for yes, gold in guesses), sum(yes and gold for yes, gold in guesses), cutoff)

    return max(THRESHOLDS, key=score)


def access_metrics(rows, flags):
    result = Counter(total=len(rows))
    groups = {}
    for row, allowed in zip(rows, flags, strict=True):
        gold = can_read(row['action'])
        result.update({'positive': gold, 'allowed_positive': gold and allowed, 'wrong_allow': allowed and not gold})
        if 'family' in row:
            groups.setdefault((row['family'], row['subject']), []).append(gold == allowed)
    result.update({'groups': len(groups), 'complete_groups': sum(all(group) for group in groups.values())})
    return result


def dialogue_metrics(rows, reads):
    metrics = summarize(rows, reads)
    metrics['false_pending'] = sum(row['action'] != Intent.ASK and row['candidate']['action'] == 'ASK' for row in rows)
    metrics['reads_after_false_pending'] = sum(item['route'] is not None for row in rows if row['action'] != Intent.ASK
                                               for item in row['followups'])
    return metrics


def check_pending_only(output):
    """Replay an alternative policy on saved decisions without retraining the heads."""
    source = output / 'result.json'
    result = json.loads(source.read_text(encoding='utf-8'))
    splits = {}
    for split, data in result['splits'].items():
        matched = data['modes']['matched']['rows']
        guarded = data['modes']['with_gate']['rows']
        rows = []
        for raw, gate in zip(matched, guarded, strict=True):
            if raw['question'] != gate['question']:
                raise ValueError('Decisions no longer refer to the same question')
            chosen = raw if raw['action_head'] == 'READ' else gate
            rows.append(chosen)
        counts = dialogue_metrics(rows, [])
        splits[split] = {key: counts[key] for key in ('decisions', 'followup_reads', 'false_pending', 'reads_after_false_pending')}
    store_json(output / 'pending-only.json', {'source_sha256': base.sha(source), 'splits': splits})
    print(json.dumps(splits), flush=True)


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    result_path, weights = args.output / 'result.json', args.output / 'heads.pt'
    prior = json.loads(result_path.read_text(encoding='utf-8')) if args.verify else None
    paths = (args.timeline, args.reader, args.version_heads, args.action, args.archived,
             base.MODEL / 'model.safetensors', base.MODEL / 'tokenizer.json', base.MODEL / 'tokenizer_config.json')
    sources = {str(path): base.sha(path) for path in paths}
    archived = json.loads(args.archived.read_text(encoding='utf-8'))
    if sources[str(args.action)] != archived['action_sha256'] or any(sources[str(path)] != archived['sources'][str(path)]
            for path in (args.timeline, args.reader, args.version_heads)):
        raise ValueError('Inputs differ from those used for the previous replay')
    if prior is not None and (sources != prior['sources'] or base.sha(weights) != prior['heads_sha256']):
        raise ValueError('Reload inputs changed')
    names = [item[1:3] for item in SUBJECTS]
    training = matched_examples(TRAIN, TRAIN_NAMES)
    calibration = matched_examples(CALIBRATION, CAL_NAMES)
    held = matched_examples(HELD, names)
    archived_cases = examples(PRIOR_HELD, names)
    rows = training + calibration + held + archived_cases
    questions = [row['question'] for row in rows]
    if len(set(questions)) != len(questions) or [row['question'] for row in archived['rows']] != [
            row['question'] for row in archived_cases]:
        raise ValueError('Inputs differ from the intended disjoint splits')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed()
    pieces, masks = clause_features(model, tokenizer, questions)
    flat = flat_features(pieces, masks)
    action = action_head(model.config.hidden_size)
    gate = VersionAccessHead(model.config.hidden_size)
    old = action_head(model.config.hidden_size)
    old.load_state_dict(torch.load(args.action, map_location='cpu', weights_only=True), strict=True)
    if prior is None:
        train_head(action, flat[:len(training)], training)
        fit_access(gate, flat[:len(training)], training)
        with torch.inference_mode():
            cutoff = calibrate(gate(flat[len(training):len(training) + len(calibration)]).sigmoid().flatten().tolist(), calibration)
        atomic_save({'action': action.state_dict(), 'gate': gate.state_dict(), 'cutoff': cutoff}, weights)
    else:
        state = torch.load(weights, map_location='cpu', weights_only=True)
        action.load_state_dict(state['action'], strict=True)
        gate.load_state_dict(state['gate'], strict=True)
        cutoff = state['cutoff']
    version = VersionRoleHead(model.config.hidden_size)
    version.load_state_dict(torch.load(args.version_heads, map_location='cpu', weights_only=True)['expanded']['weights'], strict=True)
    bank, schedules = load_schedules(args.timeline)
    reader_state = torch.load(args.reader, map_location='cpu', weights_only=True)['atomic']
    reader = VersionReplyActions(model.config.hidden_size, reader_state['scaffold_tokens'])
    reader.load_state_dict(reader_state, strict=True)
    reader.eval().to('cuda')
    evaluations = {}
    for split, cases, offset in (('held', held, len(training) + len(calibration)),
                                 ('archived', archived_cases, len(training) + len(calibration) + len(held))):
        span = slice(offset, offset + len(cases))
        with torch.inference_mode():
            raw = action.eval()(flat[span]).argmax(-1).tolist()
            previous = old.eval()(flat[span]).argmax(-1).tolist()
            scores = gate.eval()(flat[span]).sigmoid().flatten().tolist()
        if split == 'archived' and [Intent(a).name for a in previous] != [row['action_head'] for row in archived['rows']]:
            raise ValueError('The original four-way action head changed')
        plans = role_predictions(version.eval(), pieces[span], masks[span])
        scoped = [scope_version_intent(Intent(a), permits_lookup=p >= cutoff) for a, p in zip(raw, scores, strict=True)]
        modes = {}
        for mode, decisions in (('previous', previous), ('matched', raw), ('with_gate', scoped)):
            results, reads = run_dialogues(model, tokenizer, reader, bank, dict(enumerate(schedules)), cases, decisions, plans,
                                           generate_values=mode == 'with_gate')
            if split == 'archived' and mode == 'previous' and [row['candidate'] for row in results] != [
                    row['candidate'] for row in archived['rows']]:
                raise ValueError('The prior decisions changed')
            modes[mode] = {'intent': access_metrics(cases, [can_read(a) for a in decisions]),
                           'dialogue': dialogue_metrics(results, reads), 'rows': results}
        evaluations[split] = {'gate': access_metrics(cases, [p >= cutoff for p in scores]), 'probabilities': scores, 'modes': modes}
        print(split, json.dumps({mode: {'intent': data['intent'], 'dialogue': data['dialogue']} for mode, data in modes.items()}), flush=True)
    if any(base.sha(Path(path)) != digest for path, digest in sources.items()):
        raise ValueError('A protected experiment input changed')
    report = {'sources': sources, 'heads_sha256': base.sha(weights), 'cutoff': cutoff, 'training': len(training),
              'calibration': len(calibration), 'splits': evaluations}
    if prior is not None and json.loads(json.dumps(report)) != {key: prior[key] for key in report}:
        raise ValueError('Output changed after reload')
    store_json(args.output / ('reload.json' if args.verify else 'result.json'),
               {**report, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('timeline', 'reader', 'version_heads', 'action', 'archived', 'output'):
        parser.add_argument(name, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify', action='store_true')
    mode.add_argument('--pending-only', action='store_true')
    args = parser.parse_args()
    if args.pending_only:
        check_pending_only(args.output)
    else:
        main(args)
