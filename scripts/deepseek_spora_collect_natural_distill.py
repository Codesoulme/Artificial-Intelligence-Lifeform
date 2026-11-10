"""Collect resumable local teacher-token sequences for structural distillation."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from spora.language_training import atomic_save

TRAIN = ROOT/'data/chinese-foundations.json'
DEV = ROOT/'data/conversation-development-v2.json'
OUTPUT = ROOT/'runs/deepseek-natural-distill-data-20261111'
TOPICS = ('planning', 'support', 'task_breakdown')
SEEDS = (83021, 83022, 83023)
MAX_NEW = 512


def selected_prompts():
    train = json.loads(TRAIN.read_text(encoding='utf-8'))
    dev = json.loads(DEV.read_text(encoding='utf-8'))
    excluded = {turn.strip() for scenario in dev['scenarios'] for turn in scenario['turns']}
    prompts = []
    for topic in TOPICS:
        matches = [item for item in train['dialogues'] if item['topic'] == topic]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one training topic {topic}')
        prompt = matches[0]['turns'][0][0].strip()
        if prompt in excluded:
            raise ValueError('Development prompt appears in the selected training data')
        prompts.append(prompt)
    return prompts


def run():
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected teacher hash mismatch')
    prompts = selected_prompts()
    source_hash = sha(TRAIN)
    tokenizer_hashes = {name: sha(MODEL/name) for name in ('tokenizer.json', 'tokenizer_config.json')}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    rows = []
    with torch.inference_mode():
        for index, (topic, question, seed) in enumerate(zip(TOPICS, prompts, SEEDS)):
            destination = OUTPUT/f'trajectory-{index}.pt'
            expected = {'format': 'spora-natural-teacher-sequence-v1', 'weight_sha256': WEIGHT_SHA,
                        'tokenizer_sha256': tokenizer_hashes, 'source_sha256': source_hash,
                        'topic': topic, 'question': question, 'seed': seed, 'max_new_tokens': MAX_NEW,
                        'temperature': 0.6, 'top_p': 0.95}
            if destination.exists():
                payload = torch.load(destination, map_location='cpu', weights_only=True)
                if payload.get('metadata') != expected or not isinstance(payload.get('input_ids'), torch.Tensor):
                    raise ValueError(f'Existing trajectory has incompatible metadata: {destination}')
                reused = True
            else:
                prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                                       add_generation_prompt=True)
                if not prompt.rstrip().endswith('<think>'):
                    raise ValueError('Unexpected teacher template')
                inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
                torch.manual_seed(seed)
                before = time.perf_counter()
                generated = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=True,
                                           temperature=0.6, top_p=0.95, use_cache=True,
                                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                tail = generated[0, inputs['input_ids'].shape[1]:]
                text = tokenizer.decode(tail, skip_special_tokens=True)
                payload = {'metadata': expected, 'input_ids': generated[0].detach().cpu().clone(),
                           'prefix_tokens': int(inputs['input_ids'].shape[1]),
                           'generated_tokens': int(tail.numel()),
                           'reasoning_closed': '</think>' in text,
                           'ended_with_eos': bool(tail.numel() and tail[-1] == tokenizer.eos_token_id),
                           'generation_seconds': time.perf_counter()-before}
                atomic_save(payload, destination)
                reused = False
            row = {'topic': topic, 'question': question, 'seed': seed, 'file': destination.name,
                   'sha256': sha(destination), 'prefix_tokens': payload['prefix_tokens'],
                   'generated_tokens': payload['generated_tokens'], 'reasoning_closed': payload['reasoning_closed'],
                   'ended_with_eos': payload['ended_with_eos'], 'reused': reused}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ('topic', 'generated_tokens', 'reasoning_closed',
                                                   'ended_with_eos', 'reused')}, ensure_ascii=False), flush=True)
    summary = {'format': 'spora-natural-teacher-sequence-summary-v1', 'base_sha256': WEIGHT_SHA,
               'train_source_sha256': source_hash, 'dev_source_sha256': sha(DEV), 'tokenizer_sha256': tokenizer_hashes,
               'rows': rows, 'elapsed_seconds': time.perf_counter()-started,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'limitations': ['Only three project-authored training prompts.',
                               'Teacher output can be incomplete, incorrect or repetitive.',
                               '512-token cap; token sequences are local training data, not proof of quality.',
                               'Exact development prompt overlap checked; conceptual overlap remains possible.']}
    summary_path = OUTPUT/'summary.json'
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding='utf-8'))
        if [x['sha256'] for x in existing['rows']] != [x['sha256'] for x in rows]:
            raise ValueError('Existing summary disagrees with saved trajectories')
    else:
        with summary_path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    run()
