"""Compare decoder replacements on short Chinese conversations with checkable answers."""

import argparse
import copy
import re
import time
from pathlib import Path

from audit_awake_language_layers import load_layer_pair
from deepseek_output_aligned_memory import MODEL, ROOT, SOURCE, read, sha, torch, write
from scripts.deepseek_spora_memory_answer_distill import _load_candidate
from spora.structure_transfer import SPORALayerReplacement
from train_awake_partial_layer import SPORAFeedForward
from train_awake_replacement_replay import generate, switch


CANDIDATE = ROOT / 'runs/deepseek-spora-layer18-broad-pilot-20261112'
LIMIT = 24
CEVAL_SUBJECTS = ('high_school_chinese', 'logic', 'middle_school_mathematics')
CASES = (
    ('我买了八张明信片，寄出三张，还剩几张？只答数字。', '5|五',
     '又买了两张，现在手里有几张？只答数字。', '7|七'),
    ('木盒里是铃铛，铁盒里是纽扣。纽扣在哪个盒里？只答盒名。', '铁盒',
     '把两个盒里的东西对调。现在铃铛在哪个盒里？只答盒名。', '铁盒'),
    ('演出原定周六。我说错了，改到周日。只回答最终日期。', '周日|星期日|星期天|周天',
     '如果再提前两天，改到周几？只答星期。', '周五|星期五'),
    ('先列出橙色。只输出颜色名。', '橙色|橙',
     '换成蓝色，照相同格式写出。', '蓝色|蓝'),
    ('按从小到大排列 4、1、9。只写三个数字，用逗号分隔。', '1,4,9',
     '去掉最大的，余下两个数倒序排列，沿用原来的格式。', '4,1'),
    ('两本书共十四元，每本同价。每本多少元？只答数字。', '7|七',
     '按相同单价买三本，一共多少钱？只答数字。', '21|二十一'),
    ('下面两个词里哪个表示动物：书架、海豚。只输出该词。', '海豚',
     '另一个词是什么？只输出该词。', '书架'),
    ('把短语「绿色雨伞」缩成两个字，保留物品名称。', '雨伞',
     '那它是什么颜色？只答颜色名。', '绿色|绿'),
)


def encode(tokenizer, history):
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(prompt + '</think>\n\n', add_special_tokens=False)


def answer_matches(answer, expected):
    normalized = re.sub(r'\s+', '', answer).replace('，', ',').strip('。.!！')
    return re.fullmatch(expected, normalized) is not None


@torch.inference_mode()
def conversation(decoder, tokenizer, case, position_limit):
    history, turns = [], []
    for question, expected in zip(case[::2], case[1::2], strict=True):
        history.append(dict(role='user', content=question))
        prompt = encode(tokenizer, history)
        if len(prompt) + LIMIT > position_limit:
            raise ValueError(f'Conversation exceeds candidate position limit: {question}')
        tokens = generate(decoder, tokenizer, prompt, LIMIT)
        answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        history.append(dict(role='assistant', content=answer))
        turns.append(dict(question=question, expected=expected, answer=answer,
                          correct=answer_matches(answer, expected), eos=tokens[-1] == tokenizer.eos_token_id,
                          prompt_tokens=len(prompt), tokens=tokens))
    return turns


def ceval_questions(directory, tokenizer):
    import pyarrow.parquet as parquet

    labels = 'ABCD'
    answer_ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(ids) != 1 for ids in answer_ids):
        raise ValueError('Expected one token per option label')
    answer_ids = [ids[0] for ids in answer_ids]
    for subject in CEVAL_SUBJECTS:
        path = directory / f'{subject}.parquet'
        for row in parquet.read_table(path, columns=['id', 'question', *labels, 'answer']).to_pylist():
            question = '\n'.join([row['question'], *(f'{label}. {row[label]}' for label in labels),
                                  '只回答正确选项的字母。'])
            ids = encode(tokenizer, [dict(role='user', content=question)])
            ids += tokenizer.encode('答案：', add_special_tokens=False)
            if row['answer'] not in tuple(labels):
                raise ValueError(f'Missing reference label for {subject} question {row["id"]}')
            yield subject, row['id'], ids, answer_ids, row['answer']


@torch.inference_mode()
def score_ceval(decoder, cases, position_limit):
    rows = []
    for subject, index, ids, answer_ids, expected in cases:
        choice = None
        if len(ids) <= position_limit:
            inputs = torch.tensor([ids], device='cuda')
            logits = decoder(input_ids=inputs, use_cache=False, logits_to_keep=1).logits[0, -1, answer_ids]
            choice = 'ABCD'[logits.argmax().item()]
        rows.append(dict(subject=subject, id=index, prompt_tokens=len(ids), expected=expected, choice=choice))
    return rows


@torch.inference_mode()
def score_memory(decoder, tokenizer, cases):
    rows = []
    for question, expected, memory_args, _ in cases:
        prompt = encode(tokenizer, [dict(role='user', content=question)])
        inputs = torch.tensor([prompt], device='cuda')
        output = decoder.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                  do_sample=False, max_new_tokens=32, temperature=None, top_p=None,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                  **dict(memory_args, use_cache=True))
        answer = tokenizer.decode(output[0, len(prompt):], skip_special_tokens=True).strip()
        rows.append(dict(question=question, answer=answer, expected=expected, correct=expected in answer))
    return rows


def main(output, ceval=None, partial=None):
    path = CANDIDATE / 'layer-18.pt'
    sources = {source: sha(source) for source in (SOURCE, MODEL / 'model.safetensors', path)}
    if sources[path] != read(CANDIDATE / 'result.json')['candidate_sha256']:
        raise ValueError('Third-layer candidate digest mismatch')
    if ceval is not None:
        sources.update({ceval / f'{subject}.parquet': sha(ceval / f'{subject}.parquet') for subject in CEVAL_SUBJECTS})
    if partial is not None:
        sources[partial] = sha(partial)
        if read(partial.parent / 'result.json')['candidate_sha256'] != sources[partial]:
            raise ValueError('Partial feed-forward candidate digest mismatch')
    started = time.perf_counter()
    decoder, tokenizer, originals, replacements, organism, _ = load_layer_pair()
    original_18 = decoder.base.model.layers[18]
    candidate = _load_candidate(path, 18, decoder.base).requires_grad_(False).eval()
    modes = {
        'base': originals,
        'layer_6': {**originals, 6: replacements[6]},
        'layer_12': {**originals, 12: replacements[12]},
        'pair': replacements,
        'triple': {**replacements, 18: candidate},
    }
    results = {}
    if ceval is not None:
        cases = list(ceval_questions(ceval, tokenizer))
        partial_variant = None
        memory_results = {}
        if partial is not None:
            from audit_awake_feedforward_memory import history_cases

            payload = torch.load(partial, map_location='cpu', weights_only=True)
            layer_index = payload['layer']
            if layer_index not in replacements or payload['source_sha256'] != sources[SOURCE]:
                raise ValueError('Partial feed-forward candidate belongs to a different checkpoint')
            original_mlp = originals[layer_index].mlp
            student = SPORAFeedForward(copy.deepcopy(replacements[layer_index].block.ff)).requires_grad_(False).eval()
            student.load_state_dict(payload['state'], strict=True)
            other_index = next(index for index in replacements if index != layer_index)
            partial_variant = f'partial_{layer_index}_layer_{other_index}'
            modes[partial_variant] = {**replacements, layer_index: originals[layer_index]}
            _, memory_cases = history_cases(organism)
        ceval_summary = {}
        for name, layers in modes.items():
            switch(decoder, {18: original_18, **layers})
            if partial_variant is not None:
                originals[layer_index].mlp = student if name == partial_variant else original_mlp
            max_length = min((layer.max_length for layer in layers.values() if isinstance(layer, SPORALayerReplacement)),
                             default=decoder.base.config.max_position_embeddings)
            rows = score_ceval(decoder, cases, max_length)
            results[name] = rows
            if partial_variant is not None and name in ('pair', partial_variant, f'layer_{other_index}'):
                memory_rows = score_memory(decoder, tokenizer, memory_cases)
                memory_results[name] = memory_rows
                print(name, 'memory', sum(row['correct'] for row in memory_rows), '/', len(memory_rows), flush=True)
            ceval_summary[name] = {}
            for subject in CEVAL_SUBJECTS:
                subject_rows = [row for row in rows if row['subject'] == subject and row['choice'] is not None]
                shared = [row for row in subject_rows if row['prompt_tokens'] <= candidate.max_length]
                ceval_summary[name][subject] = dict(correct=sum(row['choice'] == row['expected'] for row in subject_rows),
                                                    eligible=len(subject_rows),
                                                    shared_correct=sum(row['choice'] == row['expected'] for row in shared),
                                                    shared_eligible=len(shared))
            print(name, ceval_summary[name], flush=True)
        if any(sha(source) != digest for source, digest in sources.items()):
            raise ValueError('A source changed during the audit')
        write(output, dict(summary=ceval_summary, results=results, memory=memory_results,
                           shared_position_limit=candidate.max_length,
                           sources={str(source): digest for source, digest in sources.items()},
                           seconds=time.perf_counter() - started, peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        return
    for name, layers in modes.items():
        switch(decoder, {18: original_18, **layers})
        rows = []
        for case in CASES:
            turns = conversation(decoder, tokenizer, case, candidate.max_length)
            rows.append(turns)
            print(name, len(rows), [(turn['answer'], turn['correct']) for turn in turns], flush=True)
        results[name] = rows
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('A source changed during the audit')
    summary = {
        name: dict(first_correct=sum(turns[0]['correct'] for turns in rows),
                   followup_correct=sum(turns[1]['correct'] for turns in rows),
                   both_correct=sum(all(turn['correct'] for turn in turns) for turns in rows),
                   eos=sum(turn['eos'] for turns in rows for turn in turns))
        for name, rows in results.items()
    }
    write(output, dict(summary=summary, results=results, max_new_tokens=LIMIT,
                       sources={str(path): digest for path, digest in sources.items()},
                       seconds=time.perf_counter() - started, peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ceval', type=Path, help='Score local C-Eval validation shards')
    parser.add_argument('--partial', type=Path, help='Include a saved feed-forward transplant in C-Eval')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose an unused output path')
    if args.partial is not None and args.ceval is None:
        parser.error('--partial requires --ceval')
    main(args.output, args.ceval, args.partial)
