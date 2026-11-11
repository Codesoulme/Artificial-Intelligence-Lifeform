"""Recalibrate only the output adapter on frozen cached decoder logits."""
import copy
import json
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_counterfactual_output_audit import VALUES, TEMPLATES, OUTPUT as PRIOR
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import _memory_args
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputLogitsProcessor

OUTPUT = ROOT / 'runs/deepseek-spora-output-recalibration-pilot-20261112'
TRAIN_TEMPLATES = ('{name}目前最爱哪种饮料？', '现在{name}更喜欢喝什么饮品？')
TRAIN_PERSONS = (1, 2)
EVAL_PERSONS = (0, 3)


def make_state(chat, person, value):
    state = fresh_controlled_state(chat.organism)
    for index, (key, _, default_value) in enumerate(ENROLLMENTS):
        state, decision = observe_sourced(chat.organism, state, key,
                                          value if person == index else default_value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled write rejected')
    return state


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if prior['source_checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Reference checkpoint mismatch')
    torch.manual_seed(11921)
    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        runtime.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: load_full_streamed()
        chat = runtime.load_chat()
        original_adapter = chat.decoder.output_adapter
        if original_adapter is None:
            raise ValueError('Expected an existing output adapter')
        device = next(chat.organism.parameters()).device
        cached_logits, cached_memory, targets = [], [], []
        for person in TRAIN_PERSONS:
            key, name, _ = ENROLLMENTS[person]
            for value in VALUES:
                chat.state = make_state(chat, person, value)
                key_tensor = torch.tensor([key], dtype=torch.long, device=device)
                zeros = torch.zeros_like(key_tensor)
                with torch.no_grad():
                    memory = query(chat.organism, chat.state, key_tensor, zeros, zeros)
                if int(memory.argmax(-1)[0]) != value:
                    raise ValueError('Training memory does not match written value')
                target = chat.tokenizer(WORDS[value], add_special_tokens=False)['input_ids']
                if len(target) != 1:
                    raise ValueError('Pilot requires a single-token target')
                for template in TRAIN_TEMPLATES:
                    question = template.format(name=name)
                    prefix = closed_think_inputs(chat.tokenizer, question, 'cuda')
                    args = _memory_args(chat.organism, chat.state, {'key': key, 'intent': 'current'})
                    with torch.no_grad():
                        base_logits = chat.decoder(**prefix, **args).logits[:, -1, :]
                    cached_logits.append(base_logits.detach().float().cpu())
                    cached_memory.append(memory.detach().float().cpu())
                    targets.append(target[0])
        logits = torch.cat(cached_logits)
        memory = torch.cat(cached_memory)
        target = torch.tensor(targets, dtype=torch.long)
        adapter = copy.deepcopy(original_adapter).cpu().train()
        with torch.no_grad():
            original_bias = adapter(memory).detach()
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.0001, weight_decay=0.0)
        losses = []
        for step in range(80):
            bias = adapter(memory).float()
            # Penalize changes to the existing per-example bias to limit drift;
            # this does not establish preservation outside the small train set.
            loss = F.cross_entropy(logits + bias, target) + 0.001 * (bias - original_bias).square().mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            if step in (0, 19, 39, 79):
                losses.append({'step': step + 1, 'loss': float(loss.detach())})
        adapter.eval()
        with torch.no_grad():
            train_parent = int((logits + original_bias).argmax(-1).eq(target).sum())
            train_candidate = int((logits + adapter(memory).float()).argmax(-1).eq(target).sum())
        adapter.to(device)
        rows = []
        for person in EVAL_PERSONS:
            key, name, _ = ENROLLMENTS[person]
            for value in VALUES:
                chat.state = make_state(chat, person, value)
                before = frozen_digest(chat.organism, chat.state)
                key_tensor = torch.tensor([key], dtype=torch.long, device=device)
                with torch.no_grad():
                    output = query(chat.organism, chat.state, key_tensor, torch.zeros_like(key_tensor),
                                   torch.zeros_like(key_tensor))
                memory_top = int(output.argmax(-1)[0])
                for template_index, template in enumerate(TEMPLATES):
                    question = template.format(name=name)
                    chat.decoder.output_adapter = original_adapter
                    parent = chat.answer(question)
                    chat.decoder.output_adapter = adapter
                    candidate = chat.answer(question)
                    with patch.object(MemoryOutputLogitsProcessor, '__call__', lambda self, ids, scores: scores):
                        no_bias = chat.answer(question)
                    expected = WORDS[value]
                    row = {'person': person, 'value': value, 'template': template_index,
                           'memory_top_value': memory_top, 'expected_word': expected,
                           'parent': parent, 'candidate': candidate, 'no_bias': no_bias}
                    for label in ('parent', 'candidate', 'no_bias'):
                        r = row[label]
                        row[label + '_correct'] = bool(r['accepted'] and r.get('key') == key and r.get('answer') == expected)
                    rows.append(row)
                    print(json.dumps({'person': person, 'value': value, 'template': template_index,
                                      'memory_matches': memory_top == value,
                                      **{label: row[label + '_correct'] for label in ('parent', 'candidate', 'no_bias')}}), flush=True)
                if frozen_digest(chat.organism, chat.state) != before:
                    raise ValueError('Evaluation modified temporary memory')
        chat.decoder.output_adapter = original_adapter
        adapter_state = {k: v.detach().cpu() for k, v in adapter.state_dict().items()}
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected source changed')
    summary = {'examples': len(rows), 'memory_matches': sum(r['memory_top_value'] == r['value'] for r in rows),
               **{label + '_correct': sum(r[label + '_correct'] for r in rows) for label in ('parent', 'candidate', 'no_bias')}}
    OUTPUT.mkdir(parents=False, exist_ok=False)
    adapter_file = OUTPUT / 'adapter.pt'
    torch.save({'format': 'spora-output-recalibration-development-v1', 'state': adapter_state,
                'config': original_adapter.config_dict(), 'source_checkpoint_sha256': checkpoint_sha}, adapter_file)
    result = {'status': 'experimental_adapter_calibration_not_deployed', 'summary': summary, 'rows': rows,
              'training_examples': len(targets), 'training_first_token_parent': train_parent,
              'training_first_token_candidate': train_candidate, 'losses': losses,
              'source_checkpoint_sha256': checkpoint_sha, 'adapter_sha256': runtime.sha(adapter_file),
              'seconds': time.perf_counter() - started,
              'limitations': ['Only 12 synthetic training conditions and 12 held-person/template conditions.',
                              'Evaluation conditions were used in previous analysis and are a development set.',
                              'Three finite values, manually written numerical facts, and exact one-word scoring.',
                              'Output adapter alone trained; base and individual remain unchanged.',
                              'Cached logits use oracle numerical keys and may differ from routed application.',
                              'No general conversation, natural memory writing, or production deployment tested.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
