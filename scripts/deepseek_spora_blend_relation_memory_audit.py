"""Compare language classification and counterfactual memory across graft strengths."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_spora_relation_base_comparison import OUTPUT as PRIOR, prompt_for
from scripts.deepseek_spora_sentence_lora_pilot import make_state
from spora.structure_transfer import SPORALayerBlend

OUTPUT = ROOT / 'runs/deepseek-spora-blend-relation-memory-20261114.json'
STRENGTHS = (0.0, 0.25, 0.5, 1.0)
MEMORY_CASES = ((0, 1), (0, 7), (3, 3), (3, 4))


def logits_for(base, tokenizer, question):
    inputs = closed_think_inputs(tokenizer, prompt_for(question), 'cuda')
    with torch.inference_mode():
        return base(**inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits[0, -1].float().cpu()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    prior_sha = p.sha(PRIOR)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if base_sha != p.WEIGHT_SHA or prior['base_sha256'] != base_sha or prior['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Prior provenance mismatch')
    rows = [{key: row[key] for key in ('set', 'group', 'question', 'label')} for row in prior['rows']]
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    memory_rows = []
    with tempfile.TemporaryDirectory(prefix='spora-blend-relation-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        base = p.load_full_streamed().eval()
        originals = {i: base.model.layers[i] for i in (6, 12)}
        chat = p.runtime.load_chat(base_loader=lambda: base)
        source_digest = p.frozen_digest(chat.organism, chat.state)
        replacements = {i: base.model.layers[i] for i in originals}
        states = {(i, value): make_state(chat, KEYS[i], value) for i, value in MEMORY_CASES}
        state_digests = {key: p.frozen_digest(chat.organism, state) for key, state in states.items()}
        tokens = {label: chat.tokenizer.encode(label, add_special_tokens=False) for label in ('是', '否')}
        if any(len(ids) != 1 for ids in tokens.values()):
            raise ValueError('Expected single-token labels')
        teacher = []
        for i in originals:
            base.model.layers[i] = originals[i]
        for row in rows:
            teacher.append(logits_for(base, chat.tokenizer, row['question']))
        for strength in STRENGTHS:
            for i in originals:
                base.model.layers[i] = (originals[i] if strength == 0 else replacements[i] if strength == 1 else
                                        SPORALayerBlend(originals[i], replacements[i], strength).eval())
            for row, ref in zip(rows, teacher):
                logits = logits_for(base, chat.tokenizer, row['question'])
                ref_log = F.log_softmax(ref, dim=-1)
                actual_log = F.log_softmax(logits, dim=-1)
                yes, no = float(logits[tokens['是'][0]]), float(logits[tokens['否'][0]])
                row[str(strength)] = {'accepted': yes > no, 'top_id': int(logits.argmax()),
                                      'top_token': chat.tokenizer.decode([int(logits.argmax())]),
                                      'teacher_top_id': int(ref.argmax()),
                                      'kl_teacher_to_candidate': float((ref_log.exp() * (ref_log - actual_log)).sum())}
            for index, value in MEMORY_CASES:
                state = states[index, value]
                args, evidence = p.runtime._prepare_memory_read(chat.organism, state, KEYS[index], 'current', None)
                if int(evidence.argmax(-1)[0]) != value:
                    raise ValueError('Controlled native read failed')
                question = f'{NAMES[index]}现在最喜欢喝什么饮料？'
                prompt = closed_think_inputs(chat.tokenizer, question, 'cuda')
                with torch.inference_mode():
                    generated = chat.decoder.generate(**prompt, **args, max_new_tokens=24, do_sample=False,
                                                      pad_token_id=chat.tokenizer.pad_token_id,
                                                      eos_token_id=chat.tokenizer.eos_token_id)
                tail = generated[0, prompt['input_ids'].shape[1]:].tolist()
                answer = chat.tokenizer.decode(tail, skip_special_tokens=True).strip()
                memory_rows.append({'strength': strength, 'name': NAMES[index], 'question': question,
                                    'expected': p.WORDS[value], 'answer': answer, 'token_ids': tail,
                                    'contains_target': p.WORDS[value] in answer,
                                    'exact': answer.rstrip('。.!！ ') == p.WORDS[value],
                                    'ended': chat.tokenizer.eos_token_id in tail})
            print(json.dumps({'strength': strength, 'classification_cases': len(rows),
                              'memory_cases': len(MEMORY_CASES)}), flush=True)
        for i in originals:
            base.model.layers[i] = replacements[i]
        if p.frozen_digest(chat.organism, chat.state) != source_digest or any(
                p.frozen_digest(chat.organism, state) != state_digests[key] for key, state in states.items()):
            raise ValueError('Protected state changed')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(PRIOR) != prior_sha:
        raise ValueError('Protected inputs changed')
    summary = {}
    for strength in STRENGTHS:
        label = str(strength)
        memory = [r for r in memory_rows if r['strength'] == strength]
        summary[label] = {'cases': len(rows), 'correct_choice': sum(r[label]['accepted'] == bool(r['label']) for r in rows),
                          'negative_accepted': sum(r[label]['accepted'] for r in rows if not r['label']),
                          'label_top': sum(r[label]['top_token'] in ('是', '否') for r in rows),
                          'teacher_top_agreement': sum(r[label]['top_id'] == r[label]['teacher_top_id'] for r in rows),
                          'mean_kl': sum(r[label]['kl_teacher_to_candidate'] for r in rows) / len(rows),
                          'memory_cases': len(memory), 'memory_exact': sum(r['exact'] for r in memory),
                          'memory_contains': sum(r['contains_target'] for r in memory)}
    report = {'summary': summary, 'rows': rows, 'memory_rows': memory_rows,
              'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha, 'prior_sha256': prior_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Previously inspected authored classification prompts; not an independent benchmark.',
                              'First-token KL and forced choice do not establish general language quality.',
                              'Memory comparison uses only four numeric fact writes and two question templates.',
                              'Direct numeric addresses bypass natural language routing; no output adapter enabled.',
                              'Router provenance is tied to full replacement, not blended structural variants.',
                              'Blending executes both branches and requires cache-disabled generation.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
