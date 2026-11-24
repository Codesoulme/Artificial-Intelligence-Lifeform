"""Replay mixed Chinese dialogue through the actual spora_chat.cmd entry point."""

import argparse
from collections import Counter, deque
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'runs/spora-awake-local'
INITIAL_VALUE = '苦荞茶'
CORRECTED_VALUE = '米香玄米茶'
VALUES = (INITIAL_VALUE, CORRECTED_VALUE)


def ordinary(text, *, required=(), forbidden=(), limit=None, prior=False):
    return dict(text=text, kind='ordinary', required=required, forbidden=forbidden, limit=limit, prior=prior)


def recall(text, value):
    return dict(text=text, kind='recall', expected=value)


FIRST_SESSION = (
    ordinary('写一句提醒同事清理会议室的通知，要提到椅子。', required=('椅子',)),
    ordinary('改成发给保洁的简短提醒，保留椅子，但避开“会议室”三个字。',
             required=('椅子',), forbidden=('会议室',), limit=42, prior=True),
    ordinary('把提醒改成问句，仍提到椅子，但不要恢复被删掉的词。',
             required=('椅子',), forbidden=('会议室',), limit=42, prior=True),
    dict(text=f'/记住 齐宁最喜欢喝{INITIAL_VALUE}。', kind='write'),
    recall('齐宁平常最爱喝什么？', INITIAL_VALUE),
    ordinary('继续修改刚才的提醒，把椅子换成桌子，还是别提刚才删掉的那个词。',
             required=('桌子',), forbidden=('椅子', '会议室'), limit=44, prior=True),
    ordinary('另起一个话题。给我一句约同事周四三点在东门见的短信。', required=('周四', '东门')),
    ordinary('改到周五，地点保持不变，短信里不要提周四。',
             required=('周五', '东门'), forbidden=('周四',), prior=True),
    dict(text=f'/记住 齐宁现在最喜欢喝{CORRECTED_VALUE}。', kind='write'),
    recall('齐宁平常最爱喝什么？', CORRECTED_VALUE),
    recall('齐宁上一次记录里最爱喝什么？', INITIAL_VALUE),
    ordinary('只评价这句读起来是否自然，不要回答问题。“齐宁平常最爱喝什么？”'),
    ordinary('说一句关于雨天出门的提醒，提到雨伞。', required=('雨伞',)),
    ordinary('短一些，还要提到雨伞，但避开“下雨”和“雨天”。',
             required=('雨伞',), forbidden=('下雨', '雨天'), limit=24, prior=True),
    dict(text='从来没登记过的沈砚平常爱喝什么？', kind='unknown'),
)
SECOND_SESSION = (
    recall('齐宁平常最爱喝什么？', CORRECTED_VALUE),
    recall('齐宁上一次记录里最爱喝什么？', INITIAL_VALUE),
    ordinary('一句话告诉同事，备用钥匙放在抽屉里。', required=('备用钥匙', '抽屉')),
    ordinary('改成已交给门卫，继续说的是同一把东西，别再出现抽屉。',
             required=('钥匙', '门卫'), forbidden=('抽屉',), limit=42, prior=True),
    ordinary('又改了，保留交给了谁，并在句末加“请签收”。',
             required=('门卫', '请签收'), forbidden=('抽屉',), limit=42, prior=True),
    ordinary('把“齐宁平常最爱喝什么”换一种说法，只改写这句话。'),
    ordinary('换个话题。用一句话告诉小朋友为什么泡好的茶要放一会儿再喝。'),
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def protected_profile():
    return {file.name: digest(file) for file in PROFILE.iterdir()
            if file.is_file() and file.suffix != '.lock'}


def execute(profile, cases, budget, *, ordinary_head=None):
    questions = [case['text'] for case in cases]
    command = ['cmd.exe', '/d', '/c', str(ROOT / 'spora_chat.cmd'), '--memory-file', str(profile / 'memory.sqlite3'),
               '--awake-state', str(profile / 'awake.pt'), '--json']
    if budget != 64:
        command.extend(('--ordinary-new-tokens', str(budget)))
    if ordinary_head is not None:
        command.extend(('--ordinary-intent-head', str(ordinary_head)))
    started = perf_counter()
    result = subprocess.run(command, cwd=ROOT, input='\n'.join(questions) + '\n', encoding='utf-8',
                            capture_output=True, timeout=900, check=False)
    seconds = perf_counter() - started
    if result.returncode:
        raise RuntimeError(f'Chat exited with {result.returncode}: {result.stderr}\n{result.stdout}')
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != len(cases):
        raise ValueError(f'Chat returned {len(rows)} of {len(cases)} turns.\n{result.stderr}\n{result.stdout}')
    return rows, seconds


def grade(cases, outputs):
    counts = Counter()
    rows = []
    previous = None
    for case, raw in zip(cases, outputs, strict=True):
        kind = case['kind']
        action, text = raw['action'], raw.get('response', '')
        hits = [value for value in VALUES if value in text]
        counts[kind] += 1
        evidence = {'memory_hits': hits}
        if kind == 'write':
            evidence['write_ok'] = action == 'stored'
            counts['writes_ok'] += evidence['write_ok']
        elif kind == 'recall':
            evidence['recall_ok'] = action == 'read' and hits == [case['expected']]
            counts['recall_ok'] += evidence['recall_ok']
        else:
            counts['unwanted_value'] += bool(hits)
        if kind == 'ordinary':
            limit = case['limit']
            evidence.update(required_ok=all(term in text for term in case['required']),
                            forbidden_ok=not any(term in text for term in case['forbidden']),
                            length_ok=limit is None or len(text) <= limit,
                            repeats_previous=previous is not None and text == previous,
                            history_kept=bool(raw.get('context_turns', 0)))
            evidence['objective_constraints'] = (action == 'ordinary' and bool(text) and not hits
                                                   and evidence['required_ok'] and evidence['forbidden_ok']
                                                   and evidence['length_ok'] and not evidence['repeats_previous'])
            counts['ordinary_routed'] += action == 'ordinary'
            counts['ordinary_constraints'] += evidence['objective_constraints']
            counts['ordinary_end'] += raw.get('generation', {}).get('end_token', False)
            counts['turns_requiring_history'] += case['prior']
            counts['requested_history_present'] += case['prior'] and evidence['history_kept']
        previous = text if kind == 'ordinary' else previous
        rows.append({'case': case, 'output': raw, 'evidence': evidence})
    return dict(counts), rows


def main(args):
    target = args.output.resolve()
    if target.is_relative_to(PROFILE.resolve()) or not target.is_relative_to(ROOT / 'runs'):
        raise ValueError('The audit needs a fresh directory under runs')
    target.mkdir(parents=True, exist_ok=False)
    originals = protected_profile()
    for budget in args.budget:
        profile = target / f'tokens-{budget}'
        profile.mkdir()
        first, seconds = execute(profile, FIRST_SESSION, budget)
        summary, rows = grade(FIRST_SESSION, first)
        print(budget, 'first', summary, flush=True)
        (profile / 'first.json').write_text(json.dumps({'seconds': seconds, 'summary': summary, 'rows': rows},
                                                       ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        second, seconds = execute(profile, SECOND_SESSION, budget)
        summary, rows = grade(SECOND_SESSION, second)
        print(budget, 'restart', summary, flush=True)
        (profile / 'restart.json').write_text(json.dumps({'seconds': seconds, 'summary': summary, 'rows': rows},
                                                         ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if protected_profile() != originals:
            raise ValueError('The original chat profile changed during the isolated run')


def replay_scope_fix(output):
    profile = output.resolve() / 'tokens-64'
    if not (profile / 'memory.sqlite3').is_file() or profile.is_relative_to(PROFILE.resolve()):
        raise ValueError('The scope check needs the isolated long-running chat profile')
    ordinary_start = {**FIRST_SESSION[0], 'text': '另起一个话题。' + FIRST_SESSION[0]['text']}
    cases = SECOND_SESSION + (FIRST_SESSION[11], ordinary_start, FIRST_SESSION[1],
                              recall('齐宁平常最爱喝什么？', CORRECTED_VALUE), FIRST_SESSION[5])
    originals = protected_profile()
    checkpoint = ROOT / 'runs/deepseek-awake-ordinary-intent.pt'
    for mode, head in (('quote-fix', None), ('quote-fix-with-ordinary-head', checkpoint)):
        results, seconds = execute(profile, cases, 64, ordinary_head=head)
        summary, rows = grade(cases, results)
        with (profile / f'{mode}.json').open('x', encoding='utf-8') as stream:
            json.dump({'seconds': seconds, 'summary': summary, 'rows': rows}, stream, ensure_ascii=False, indent=2)
        print(mode, summary, flush=True)
        if protected_profile() != originals:
            raise ValueError('The original chat profile changed during the scope check')


def base_replay(output, budget):
    import torch

    from scripts.deepseek_addressed_read import closed_think_inputs
    from scripts.deepseek_local_baseline import MODEL, AutoTokenizer, sha
    from scripts.deepseek_spora_full_stream_loader import load_full_streamed

    target = output / f'tokens-{budget}'
    path = target / 'base-replay.json'
    sources = {phase: digest(target / f'{phase}.json') for phase in ('first', 'restart')}
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    model = load_full_streamed()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    report = {}
    for phase in sources:
        logged = json.loads((target / f'{phase}.json').read_text(encoding='utf-8'))
        history = deque(maxlen=4)
        responses = []
        compared = Counter()
        for row in logged['rows']:
            case, original = row['case'], row['output']
            if original['action'] != 'ordinary':
                history.clear()
                responses.append(original)
                continue
            retained = original['context_turns']
            previous = list(history)[len(history) - retained:]
            messages = [{'role': role, 'content': content} for user, assistant in previous
                        for role, content in (('user', user), ('assistant', assistant))]
            question = case['text']
            inputs = closed_think_inputs(tokenizer, question, 'cuda', history=messages)
            length = int(inputs['input_ids'].shape[-1])
            if length != original['generation']['prompt_tokens'] or len(previous) != retained:
                raise ValueError('Could not reconstruct the production prompt')
            with torch.inference_mode():
                ids = model.generate(**inputs, do_sample=False, max_new_tokens=original['generation']['budget'],
                                     pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)[0, length:]
            text = tokenizer.decode(ids, skip_special_tokens=True).strip()
            history.append((question, original['response']))
            compared.update({'turns': 1, 'exact_response_matches': text == original['response']})
            responses.append({**original, 'response': text, 'generation': {**original['generation'],
                              'tokens': len(ids), 'end_token': bool(len(ids) and int(ids[-1]) == tokenizer.eos_token_id)}})
        summary, rows = grade([row['case'] for row in logged['rows']], responses)
        report[phase] = {'summary': summary, 'compared': compared, 'rows': rows}
        print('base', phase, compared, summary, flush=True)
    with path.open('x', encoding='utf-8') as file:
        json.dump({'seconds': perf_counter() - started, 'sources': sources, 'base_sha256': sha(MODEL / 'model.safetensors'),
                   'peak_bytes': torch.cuda.max_memory_allocated(), 'phases': report}, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--budget', type=int, nargs='+', default=(64, 128))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--base-replay', type=int, metavar='BUDGET', help='Use the actual runtime histories with the original decoder')
    mode.add_argument('--recheck-scope', action='store_true', help='Recheck quoted requests after installing the scope fix')
    cli = parser.parse_args()
    if not shutil.which('cmd.exe'):
        parser.error('The audit requires the Windows launcher')
    if cli.base_replay is not None:
        base_replay(cli.output, cli.base_replay)
    elif cli.recheck_scope:
        replay_scope_fix(cli.output)
    else:
        main(cli)
