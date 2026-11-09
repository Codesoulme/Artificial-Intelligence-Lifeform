"""Learn finite SPORA addresses from frozen DeepSeek hidden states without P26 at inference.

All 64 entity/address pairs are supervised. Held-out questions and trajectories
test phrasing/state transfer, not generalization to unknown entities.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_full_answer import thinking_inputs
from scripts.deepseek_local_baseline import MODEL, REVISION, WEIGHT_SHA
from scripts.open_base_addressed_alignment import TRAIN_PROMPTS, HELD_PROMPTS, QUERY_SHA
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_real_memory_alignment import native_read
from scripts.open_base_trajectory_comparison import build_trajectories, numeric_correct
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks

EXPANDED_PROMPTS = TRAIN_PROMPTS + (
    '对象{key}目前的值是什么？', '请说出对象{key}当前的数值。', '对象{key}的值是几？',
    '编号{key}当前是多少？', '请读取对象{key}当前的数值。', '查询对象{key}当前的值。',
    '对象{key}现在记录的数值是什么？', '编号{key}的当前数值是几？',
    '对象{key}目前记录的数值是？', '请回答对象{key}当前的值。',
    '对象{key}现在记录了什么数值？', '编号{key}目前对应的值是什么？',
    '请给出对象{key}的当前值。', '对象{key}当前数值为多少？')
NOVEL_PROMPTS = ('帮我查一下对象{key}的当前数值。', '对象{key}当前记录的数字是多少？')


def select_addresses(adapter, features):
    if features.ndim != 3 or features.shape[1] != 2 or features.shape[2] != adapter.hidden_width:
        raise ValueError('Expected features with shape [batch, 2, hidden_width]')
    return [F.normalize(adapter.query[site](features[:, site].float()), dim=-1) for site in range(2)]


@torch.no_grad()
def extract(model, tokenizer, prompts, device, batch_size=8):
    features = []
    for offset in range(0, len(prompts), batch_size):
        questions = prompts[offset:offset + batch_size]
        texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                                add_generation_prompt=True) for q in questions]
        if any('<think>' not in text or '</think>' in text for text in texts):
            raise ValueError('Unexpected reasoning template')
        inputs = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
        lengths = inputs['attention_mask'].sum(-1) - 1
        output = model(**inputs, use_cache=False, output_hidden_states=True)
        row = torch.arange(len(questions), device=device)
        # Hidden-state index zero is the embedding; 6 and 18 are the outputs of layers 5 and 17.
        batch = torch.stack([output.hidden_states[index][row, lengths] for index in (6, 18)], dim=1)
        if not bool(torch.isfinite(batch).all()):
            raise ValueError('Nonfinite model features')
        features.append(batch.float().cpu())
        if (offset // batch_size + 1) % 16 == 0:
            print(f'extracted {min(offset + batch_size, len(prompts))}/{len(prompts)}', flush=True)
    return torch.cat(features)


@torch.no_grad()
def address_stats(adapter, features, labels, bases):
    q = select_addresses(adapter, features)
    predicted = torch.stack([(q[i] @ bases[i].T).argmax(-1) for i in range(2)], dim=1)
    correct = predicted == labels[:, None]
    return {'examples': len(labels), 'site_correct': correct.sum(0).tolist(),
            'both_correct': int(correct.all(-1).sum()), 'predicted': predicted.cpu().tolist()}


def run(output, source, steps, expanded=False):
    output, source = Path(output).resolve(), Path(source).resolve()
    runs = (ROOT / 'runs').resolve()
    if (not output.is_relative_to(runs) or output == runs or output.exists()
            or not source.is_relative_to(runs) or type(steps) is not int or not 1 <= steps <= 5000):
        raise ValueError('Expected new output and source under runs, with valid steps')
    source_sha = sha(source)
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') not in ('deepseek-spora-reasoning-full-answer-v1',
                                         'deepseek-spora-learned-finite-address-v1')
            or checkpoint.get('weight_sha256') != WEIGHT_SHA or checkpoint.get('individual_sha256') != INDIVIDUAL_SHA
            or (checkpoint.get('format') == 'deepseek-spora-reasoning-full-answer-v1'
                and checkpoint.get('query_sha256') != QUERY_SHA) or checkpoint.get('layers') != [5, 17]
            or checkpoint.get('current_only') is not True):
        raise ValueError('Source adapter provenance mismatch')
    def verify():
        return (sha(MODEL / 'model.safetensors') == WEIGHT_SHA and sha(INDIVIDUAL) == INDIVIDUAL_SHA
                and sha(source) == source_sha)
    if not verify():
        raise ValueError('Protected source mismatch')
    torch.manual_seed(4607)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    keys = list(range(organism.config.key_count))
    train_templates = EXPANDED_PROMPTS if expanded else TRAIN_PROMPTS
    if set(train_templates) & (set(HELD_PROMPTS) | set(NOVEL_PROMPTS)):
        raise ValueError('Train and held question templates overlap')
    train_questions = [template.format(key=key) for key in keys for template in train_templates]
    held_questions = [template.format(key=key) for key in keys for template in HELD_PROMPTS]
    novel_questions = [template.format(key=key) for key in keys for template in NOVEL_PROMPTS]
    train_labels = torch.tensor([key for key in keys for _ in train_templates], device='cuda')
    held_labels = torch.tensor([key for key in keys for _ in HELD_PROMPTS], device='cuda')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda')
    adapter.load_state_dict(checkpoint['state'], strict=True)
    adapter.requires_grad_(False)
    adapter.query.requires_grad_(True)
    train_features = extract(model, tokenizer, train_questions, 'cuda').to('cuda')
    held_features = extract(model, tokenizer, held_questions, 'cuda').to('cuda')
    novel_features = extract(model, tokenizer, novel_questions, 'cuda').to('cuda')
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    optimizer = torch.optim.AdamW(adapter.query.parameters(), lr=0.002, weight_decay=0.)
    generator = torch.Generator(device='cpu').manual_seed(4607)
    log = []
    for step in range(steps):
        indices = torch.randint(len(train_labels), (64,), generator=generator).to('cuda')
        addresses = select_addresses(adapter, train_features[indices])
        loss = sum(F.cross_entropy(10 * addresses[i] @ bases[i].T, train_labels[indices]) for i in range(2)) / 2
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite address loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapter.query.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            entry = {'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)}
            log.append(entry)
            print(json.dumps(entry), flush=True)
    adapter.eval()
    train = address_stats(adapter, train_features, train_labels, bases)
    held = address_stats(adapter, held_features, held_labels, bases)
    novel = address_stats(adapter, novel_features, held_labels, bases)
    if any(p.grad is not None for p in model.parameters()) or frozen_digest(organism, original) != before or not verify():
        raise ValueError('Protected inputs changed')
    output.mkdir(parents=False, exist_ok=False)
    snapshot = output / 'adapter.pt'
    torch.save({'format': 'deepseek-spora-learned-finite-address-v1',
                'state': {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                'source_adapter_sha256': source_sha, 'layers': [5, 17], 'current_only': True}, snapshot)
    print('address checkpoint saved; checking trajectories', flush=True)
    trajectories = build_trajectories(organism, original)
    rows = []
    with torch.inference_mode():
        for item in trajectories:
            for key in item['keys']:
                question = HELD_PROMPTS[0].format(key=key)
                inputs = thinking_inputs(tokenizer, question, 'cuda')
                addresses = select_addresses(adapter, held_features[key * len(HELD_PROMPTS):key * len(HELD_PROMPTS) + 1])
                predicted = [int((addresses[i] @ bases[i].T).argmax(-1)) for i in range(2)]
                with memory_hooks([model.model.layers[5], model.model.layers[17]], adapter, item['state'],
                                  current_only=True, addresses=addresses):
                    generated = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                               pad_token_id=tokenizer.eos_token_id)
                raw = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                final = raw.split('</think>', 1)[1].strip() if '</think>' in raw else None
                expected = item['targets'][key]
                native = native_read(organism, item['state'], [key])[0]
                rows.append({'trajectory': item['id'], 'key': key, 'question': question, 'expected': expected,
                             'native': native, 'predicted_addresses': predicted, 'raw': raw, 'final_answer': final,
                             'correct': final is not None and numeric_correct(final, expected)})
            print(f"trajectory {item['id'] + 1}/4 complete", flush=True)
    if frozen_digest(organism, original) != before or not verify():
        raise ValueError('Protected input changed during evaluation')
    report = {'status': 'finite_supervised_address_development_only', 'revision': REVISION,
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA, 'source_adapter_sha256': source_sha,
              'adapter_sha256': sha(snapshot), 'steps': steps, 'train_templates': train_templates,
              'held_templates': HELD_PROMPTS, 'novel_templates': NOVEL_PROMPTS,
              'train': train, 'held_questions': held, 'novel_questions': novel,
              'trajectory': {'examples': len(rows), 'correct': sum(row['correct'] for row in rows), 'rows': rows},
              'log': log, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['All 64 entity-to-address mappings are explicitly supervised.',
                              'Random address basis is individual-specific; no unseen-entity generalization.',
                              'Model features are extracted without memory and used as fixed addresses during generation.',
                              'P26 was used to train the source readout, but not invoked by this selector.',
                              'All trajectories share one original background and were used in earlier development.',
                              'Only finite numbered questions; unknown rejection and open semantics are untested.',
                              'Short fixed reasoning texts were supervised in the source adapter; no reasoning claim.']}
    with (output / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'train': {k: v for k, v in train.items() if k != 'predicted'},
                      'held_questions': {k: v for k, v in held.items() if k != 'predicted'},
                      'novel_questions': {k: v for k, v in novel.items() if k != 'predicted'},
                      'trajectory': {'examples': len(rows), 'correct': sum(row['correct'] for row in rows)}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--expanded', action='store_true')
    args = parser.parse_args()
    run(args.output, args.source, args.steps, args.expanded)
