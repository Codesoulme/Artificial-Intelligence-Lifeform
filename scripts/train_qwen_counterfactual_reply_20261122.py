"""Train whole-value reply conditioning on counterfactual SPORA memory reads."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_runtime_chat import _prepare_memory_read
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced
from spora.language_training import atomic_save, frozen_digest
from spora.memory_output import MemoryOutputAdapter, MemorySequenceLogitsProcessor
from transformers import AutoTokenizer

FORMAT = 'spora-counterfactual-reply-v1'
TRAIN = (
    ('陈鹭', '请核对{person}当前登记的饮品。只说饮品名字。', '这次修订前，{person}登记了哪种饮品？'),
    ('黎斐', '更新后{person}现用哪一项饮料？', '回到改动尚未发生时，{person}选的是什么？'),
    ('温禾', '只看现在生效的记录，{person}最常喝什么？', '旧便签上{person}要的饮料是什么？'),
    ('严迅', '照有效清单说说{person}的饮品选择。', '上次换过以前，{person}平常喝哪一种？'),
)
HELD = (
    ('翁溪', '给{person}准备什么喝的才符合今天的资料？', '在后来那次更正发生前，{person}喝什么？'),
    ('孟澜', '现行表格关于{person}的那项喝的写了什么？', '原件上{person}留的是哪样饮品？'),
)
KEY = 9


def records(organism, templates):
    rows = []
    for person, current, previous in templates:
        for value in range(len(WORDS)):
            state = fresh_controlled_state(organism)
            event = state.cursor
            new_value = (value + 3) % len(WORDS)
            for word in (value, new_value):
                state, decision = observe_sourced(organism, state, KEY, word, 'direct', 1.)
                if not decision['current_write']:
                    raise ValueError('Counterfactual write did not commit')
            rows.extend((question.format(person=person), state, mode, when, target)
                        for question, mode, when, target in ((current, 'current', None, new_value),
                                                            (previous, 'past', event, value)))
    return rows


def prepare(organism, row):
    text, state, mode, event, target = row
    controls, evidence = _prepare_memory_read(organism, state, KEY, mode, event)
    if int(evidence.argmax(-1)[0]) != target:
        raise ValueError('Training or evaluation target disagrees with the neural memory')
    return text, controls, evidence, target


def make_adapters(vocab_size, value_count, steps):
    return nn.ModuleList(MemoryOutputAdapter(value_count, vocab_size, rank=8) for _ in range(steps))


def sequence_processor(adapters, evidence, prefix_length):
    bias = torch.stack([adapter(evidence)[0] for adapter in adapters])
    return MemorySequenceLogitsProcessor(bias, prefix_length)


def load_reply_head(folder, checkpoint_sha, device):
    path = folder / 'reply.pt'
    report = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    digest = sha(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (digest != report['reply_sha256'] or payload['format'] != FORMAT
            or payload['checkpoint_sha256'] != checkpoint_sha):
        raise ValueError('Output head does not match the source checkpoint')
    adapters = make_adapters(payload['vocab_size'], payload['value_count'], payload['steps'])
    adapters.load_state_dict(payload['state'], strict=True)
    return adapters.to(device).eval().requires_grad_(False), digest


@torch.no_grad()
def cache(decoder, tokenizer, organism, rows, targets, *, encode=closed_think_inputs):
    logits, evidence_rows, labels = [], [], []
    for text, controls, evidence, target in (prepare(organism, row) for row in rows):
        prefix = encode(tokenizer, text, 'cuda')['input_ids']
        reference = targets[target]
        suffix = prefix.new_tensor([reference[:-1]])
        full = torch.cat((prefix, suffix), dim=1)
        output = decoder(input_ids=full, attention_mask=torch.ones_like(full), logits_to_keep=len(reference), **controls)
        logits.append(output.logits[0].cpu().half())
        labels.append(reference)
        evidence_rows.append(evidence.cpu().float())
    return logits, torch.cat(evidence_rows), labels


def fit(adapters, logits, evidence, labels, updates):
    optimizer = torch.optim.AdamW(adapters.parameters(), lr=.007, weight_decay=.001)
    rng = random.Random(20261122)
    trace = []
    for step in range(updates):
        sample = rng.sample(range(len(labels)), 8)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for slot, adapter in enumerate(adapters):
            batch = [i for i in sample if slot < len(labels[i])]
            if batch:
                bias = adapter(evidence[batch].to('cuda'))
                base = torch.stack([logits[i][slot] for i in batch]).to('cuda', dtype=torch.float32)
                target = torch.tensor([labels[i][slot] for i in batch], device='cuda')
                losses.append(F.cross_entropy(base + bias, target))
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite reply output loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapters.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 48 == 0 or step + 1 == updates:
            trace.append(float(loss.detach()))
            print('Reply update', step + 1, trace[-1], flush=True)
    return trace


@torch.inference_mode()
def evaluate(decoder, tokenizer, organism, rows, adapters=None, *, encode=closed_think_inputs, eos=None):
    eos = [tokenizer.eos_token_id] if eos is None else eos
    answered = []
    for row in rows:
        text, controls, evidence, target = prepare(organism, row)
        inputs = encode(tokenizer, text, 'cuda')
        prefix_length = inputs['input_ids'].shape[-1]
        output_args = ({'memory_output_logits': evidence} if adapters is None else
                       {'logits_processor': [sequence_processor(adapters, evidence, prefix_length)]})
        ids = decoder.generate(**inputs, **controls, **output_args, max_new_tokens=16, do_sample=False,
                               temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id,
                               eos_token_id=eos)
        tokens = ids[0, prefix_length:].tolist()
        answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        expected = WORDS[target]
        answered.append({'question': text, 'expected': expected, 'answer': answer,
                         'exact': answer.rstrip('。！ ') == expected, 'contains': expected in answer,
                         'natural_end': bool(tokens and tokens[-1] in eos)})
    by_question = {}
    for row in answered:
        by_question.setdefault(row['question'], []).append(row)
    return {'summary': {'count': len(answered), 'exact': sum(row['exact'] for row in answered),
                        'contains': sum(row['contains'] for row in answered),
                        'natural_end': sum(row['natural_end'] for row in answered),
                        'all_values_by_question': sum(all(r['exact'] for r in batch) for batch in by_question.values()),
                        'questions': len(by_question)}, 'rows': answered}


def main(checkpoint, output, updates, reference_adapter=None):
    if output.exists():
        raise FileExistsError(output)
    digest = sha(checkpoint)
    torch.manual_seed(20261122)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder, organism, original = load_integrated(checkpoint, load_full_streamed(), MODEL)
    decoder.eval().requires_grad_(False)
    state_digest = frozen_digest(organism, original)
    targets = [tokenizer.encode(word, add_special_tokens=False) + [tokenizer.eos_token_id] for word in WORDS]
    training, held = records(organism, TRAIN), records(organism, HELD)
    if {row[0] for row in training} & {row[0] for row in held}:
        raise ValueError('Reply training and evaluation questions overlap')
    logits, evidence, labels = cache(decoder, tokenizer, organism, training, targets)
    vocab, values, steps = decoder.base.config.vocab_size, organism.config.value_count, max(map(len, targets))
    adapters = make_adapters(vocab, values, steps).to('cuda')
    trace = fit(adapters, logits, evidence, labels, updates)
    del logits, evidence
    adapters.eval().requires_grad_(False)
    results = {'embedded': evaluate(decoder, tokenizer, organism, held),
               'sequence': evaluate(decoder, tokenizer, organism, held, adapters)}
    reference_sha = None
    if reference_adapter is not None:
        path = reference_adapter / 'output-adapter.pt'
        reference_sha = sha(path)
        ref_report = json.loads((reference_adapter / 'result.json').read_text(encoding='utf-8'))
        reference = torch.load(path, map_location='cpu', weights_only=True)
        if reference_sha != ref_report['output_fit']['adapter_sha256'] or reference['checkpoint_sha256'] != digest:
            raise ValueError('Reference adapter does not match the source checkpoint')
        original_adapter = decoder.output_adapter
        reference_adapter = MemoryOutputAdapter(**reference['config'])
        reference_adapter.load_state_dict(reference['state'], strict=True)
        decoder.output_adapter = reference_adapter.cuda().eval().requires_grad_(False)
        try:
            results['calibrated_first_token'] = evaluate(decoder, tokenizer, organism, held)
        finally:
            decoder.output_adapter = original_adapter
    if sha(checkpoint) != digest or frozen_digest(organism, original) != state_digest:
        raise ValueError('Original individual changed during the reply experiment')
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'reply.pt'
    atomic_save({'format': FORMAT, 'checkpoint_sha256': digest, 'vocab_size': vocab,
                 'value_count': values, 'steps': steps, 'state': adapters.cpu().state_dict()}, path)
    report = {'reply_sha256': sha(path), 'checkpoint_sha256': digest, 'train_rows': len(training),
              'updates': updates, 'loss': trace, 'reference_adapter_sha256': reference_sha, 'results': results}
    (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print({name: result['summary'] for name, result in results.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=192)
    parser.add_argument('--reference-adapter', type=Path)
    args = parser.parse_args()
    if args.updates < 1:
        parser.error('--updates must be positive')
    main(args.checkpoint, args.output, args.updates, args.reference_adapter)
