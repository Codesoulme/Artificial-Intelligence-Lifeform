"""Train a frozen-backbone action head to express versioned fields in ordinary Chinese."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_composition_training import encode as encode_actions
from scripts.audit_spora_dual_version_reply import SEED, fit
from scripts.audit_spora_revision_version_state import (
    FIELDS, HELD, MODEL, NEW_DAY, OLD_DAY, SLOT_TYPES, SOURCE, TRANSFER, WRITER_SOURCE,
    AutoTokenizer, VersionQuery, VersionReplyActions, VersionStore, WorkingTurnState, compress_dimension,
    effective_state, evaluate, extend_positions, frozen_digest, generate_versions, load_full_streamed,
    load_integrated_runtime, load_recurrent_replacements, records, retained_cases, settings_for, sha,
    store_json, using,
)
from scripts.benchmark_spora_recurrent_dialogue import encode as encode_prompt, exchange
from scripts.train_spora_recurrent_long_alignment import CORPUS, MANIFEST, material
from spora.language_training import atomic_save


# These requests and values are disjoint from the long revision evaluation cases.
TRAIN = (
    (('给巡展小队写一句调度。标牌改去哪里，由谁取回？', '麻烦提醒撤展组，写清牌子放哪儿和谁收。'),
     ('请把标牌放在', 0, '，散场时由', 1, '收回。'),
     (('云杉北台', '苏原真'), ('雨檐石阶', '阮和鸣'), ('藤廊侧门', '白简舟'))),
    (('请发一条消息，录节目几点开始，以及哪位负责剪开头？', '广播组需要一句安排，包括新录制时间和剪辑员。'),
     ('录制改在', 0, '，片头由', 1, '剪辑。'),
     (('周四早晨', '庄岸青'), ('周六晚上', '柯其南'), ('周日午间', '龚雁明'))),
    (('图书角的标签用什么做，暂存的新书放哪儿？', '请给借书处一个提醒，说明标签材料和书册存放处。'),
     ('请用', 0, '做标签，待登记书放在', 1, '。'),
     (('红边卡纸', '腰高书架'), ('牛皮纸签', '东窗竹篮'), ('白色贴片', '矮木抽屉'))),
    (('把数目和收货人合成一句，托店员交样张。', '写一句给送印师傅的交接话，谁收、交多少都提到。'),
     ('请把', 0, '样张交给', 1, '。'),
     (('九份', '易静秋'), ('七份', '卢远枝'), ('三十一份', '沙晚柠'))),
    (('请说清谁来收走招牌，另外提醒大家最终摆在哪里。', '给清场同事发信，先说收牌的人，再说位置。'),
     ('请', 1, '收回放在', 0, '的招牌。'),
     (('钟楼后门', '应池湾'), ('红砖拱廊', '祝新禹'), ('石狮旁边', '倪冬柚'))),
    (('告诉编辑谁负责剪辑开场，再补一句今天什么时候录。', '通知广播播音员，剪片头的人和录制时间都要有。'),
     ('由', 1, '负责剪片头，录音时间是', 0, '。'),
     (('晚上六点', '江寒岫'), ('上午九点', '顾初谷'), ('星期四午间', '莫玉杉'))),
)
USE = {
    'market-sign': (r'(?:放|摆)(?:到|在|至)?{new}|在{new}(?:处)?(?:放|摆)',
                    r'(?:由|请){keep}(?:负责)?(?:收|取回)|(?:收牌人|回收者)(?:是|为){keep}'),
    'radio-rehearsal': (r'录(?:音|制)(?:时间)?(?:是|改在|安排在|定在|在)?{new}|{new}(?:开始)?录(?:音|制)',
                        r'(?:由|请){keep}(?:负责)?剪(?:辑)?(?:片头|开头)|片头由{keep}(?:负责)?剪'),
    'library-storage': (r'(?:用|拿){new}(?:做|制作|当)?标签',
                        r'(?:书|书册)(?:先)?(?:放|存)(?:在|到)?{keep}|到{keep}(?:去)?找'),
    'printing-handoff': (r'{new}(?:的)?样张|样张(?:共|有|给出)?{new}',
                         r'(?:交|递|送)给{keep}|由{keep}(?:来)?接收'),
}


def training(tokenizer, held):
    forbidden = {value for case in held for value in case['values'].values()}
    questions = {case['question'] for case in held}
    passages = material(tokenizer, CORPUS, 24)
    rows, histories = [], []
    for forms, answer, values in TRAIN:
        for question in forms:
            if question in questions:
                raise ValueError('Composition lesson repeats a held question')
            for pair in values:
                if forbidden.intersection(pair) or any(value in question for value in pair):
                    raise ValueError('Composition lesson overlaps held values or leaks a field into its question')
                row = encode_actions(tokenizer, question, answer, pair, '')
                history = []
                if len(rows) % 2 == 0:
                    start = len(rows) % len(passages)
                    for text in (passages * 2)[start:start + 8]:
                        if any(value in text for value in pair):
                            raise ValueError('A distractor passage contains its training answer')
                        exchange(history, text, '好，我看到了。')
                rows.append(row)
                histories.append(history)
    return rows, histories


def grade_use(case, variants):
    for label in ('current', 'old'):
        row = variants[label]
        value = case['values']['old' if label == 'old' else 'new']
        text = row['answer']
        fields = dict(new=re.escape(value), keep=re.escape(case['values']['keep']))
        row['use'] = (row['passed'] and text.count(value) == text.count(case['values']['keep']) == 1
                      and all(re.search(pattern.format_map(fields), text) for pattern in USE[case['case']])
                      and not re.search(r'[。！？，、]{2,}', text))


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    database = args.reference.parent / 'versions.sqlite'
    writer = WRITER_SOURCE / 'adapters.pt'
    inputs = (args.checkpoint, args.reference, database, writer, SOURCE, TRANSFER, CORPUS, MANIFEST,
              *(folder / name for folder in args.layers for name in ('attention.pt', 'comparison.json')))
    protected = {file: sha(file) for file in inputs}
    original = json.loads(args.reference.read_text(encoding='utf-8'))
    if any(original['sources'].get(str(file)) != protected[file] for file in (args.checkpoint, SOURCE, TRANSFER, writer,
            *(folder / name for folder in args.layers for name in ('attention.pt', 'comparison.json')))):
        raise ValueError('Composition comparison uses different version inputs or model layers')
    if json.loads(MANIFEST.read_text(encoding='utf-8'))['files'][CORPUS.name] != protected[CORPUS]:
        raise ValueError('Training corpus digest differs')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    cases, _ = retained_cases(tokenizer)
    rows, histories = training(tokenizer, cases)
    lengths = [len(encode_prompt(tokenizer, [*history, {'role': 'user', 'content': row[0]}]))
               for row, history in zip(rows, histories, strict=True)]
    if max(length + len(row[1]) for length, row in zip(lengths, rows, strict=True)) > 2048:
        raise ValueError('Composition training exceeds the position table')
    scaffold = sorted({token for _, _, actions, _ in rows for kind, token in actions if kind == 'text'})
    model = load_full_streamed(stage_output_projection=True)
    rotary = model.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, model, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    model.lm_head.to(model.device)
    identity = frozen_digest(runtime.organism, runtime.state)
    if identity != original['individual_sha256']:
        raise ValueError('Composition uses a different individual')
    replacements = load_recurrent_replacements(decoder, args.layers, protected[args.checkpoint])
    if set(replacements) != {19, 22}:
        raise ValueError('Expected the two-layer recurrent baseline')
    embedding = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    binding = sha256(json.dumps((identity, protected[writer], protected[args.checkpoint])).encode()).hexdigest()
    settings = settings_for(decoder, tokenizer, 48) | dict(require_coverage=True, complete_on_eos=True,
                                                           question_only=False)
    output = {}
    with VersionStore(database, binding) as store, using(replacements, 'recurrent'):
        catalog = store.catalog()
        states = []
        for case in cases:
            bank = {'current': effective_state(store, catalog, codebook, case, VersionQuery('current')),
                    'old': effective_state(store, catalog, codebook, case, VersionQuery('as_of', OLD_DAY)),
                    'empty': WorkingTurnState()}
            for label, key in (('current', 'new'), ('old', 'old')):
                decoded = tuple(tokenizer.decode(bank[label].slots[SLOT_TYPES.index(field)])
                                for field in FIELDS[case['case']])
                if decoded != (case['values'][key], case['values']['keep']):
                    raise ValueError('Version store returned unexpected field values')
            states.append(bank)
        for label, context in (('short', None), ('mixed', histories)):
            torch.manual_seed(SEED)
            head = VersionReplyActions(model.config.hidden_size, scaffold, slots=2)
            fit(model, tokenizer, head, rows, args.steps, histories=context)
            artifact = args.output / f'{label}.pt'
            atomic_save(head.state_dict(), artifact)
            head.load_state_dict(torch.load(artifact, map_location='cpu', weights_only=True), strict=True)
            head.to(model.device).eval().requires_grad_(False)
            result = []
            for case, bank, prior in zip(cases, states, original['rows'], strict=True):
                answers = generate_versions(decoder, tokenizer, settings, head, bank, case)
                if ((case['case'], case['variant']) != (prior['case'], prior['variant'])
                        or answers['empty']['answer'] != prior['answers']['coverage']['empty']['answer']):
                    raise ValueError('The empty-state result changed ordinary generation')
                grade_use(case, answers)
                result.append(dict(case=case['case'], variant=case['variant'], values=case['values'], answers=answers))
            summary = {version: {metric: sum(item['answers'][version][metric] for item in result)
                                 for metric in ('passed', 'use', 'eos')} for version in ('current', 'old')}
            summary['paired_use'] = sum(item['answers']['current']['use'] and item['answers']['old']['use']
                                        for item in result)
            output[label] = dict(rows=result, summary=summary, head_sha256=sha(artifact))
            store_json(args.output / f'{label}.json', output[label])
            print(label, summary, flush=True)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))['summary']
        if memory != original['memory']:
            raise RuntimeError('Addressed memory differs from the matched baseline')
    if identity != frozen_digest(runtime.organism, runtime.state) or any(
            sha(file) != digest for file, digest in protected.items()):
        raise RuntimeError('A frozen source or individual changed')
    store_json(args.output / 'result.json', dict(results=output, training_rows=len(rows), steps=args.steps,
               mixed_input_lengths=lengths, individual_sha256=identity, memory=memory,
               sources={str(file): value for file, value in protected.items()}, seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('reference', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('layers', nargs=2, type=Path)
    cli.add_argument('--steps', type=int, default=240)
    args = cli.parse_args()
    if args.steps < 1:
        cli.error('Expected a positive training step count')
    main(args)
