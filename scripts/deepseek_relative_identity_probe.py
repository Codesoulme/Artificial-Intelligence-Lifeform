"""Research relative speaker identity in an isolated finite-memory setting."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_dulemon_routing_audit import ARCHIVE, ARCHIVE_SHA, sample_archive
from scripts.deepseek_integrated_checkpoint_probe import OUTPUT as CHECKPOINT
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_observation_factorized_probe import FORMAT as VALUE_FORMAT, FactorizedProbe, combined_logits
from scripts.deepseek_observation_value_probe import ABSTAIN, WORDS, pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_semantic_value_pilot import word_correct
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.current_memory_status import current_memory_status
from spora.growing import query
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-relative-identity-two-callers-20261110'
VALUE = ROOT / 'runs/deepseek-observation-factorized-20261110'
SPEAKER, ADDRESSEE, UNKNOWN = 0, 1, 2
TRAIN = {
    SPEAKER: ('我喜欢喝{word}。', '我最喜欢的饮料是{word}。', '我不爱喝{word}。',
              '我的饮品偏好是{word}。', '我爱喝什么？', '我最喜欢什么饮料？'),
    ADDRESSEE: ('你喜欢喝{word}。', '你最喜欢的饮料是{word}。', '你不爱喝{word}。',
                '你的饮品偏好是{word}。', '你爱喝什么？', '你最喜欢什么饮料？'),
    UNKNOWN: ('我和你都喜欢喝{word}。', '他说：“我喜欢喝{word}。”',
              '小林喜欢喝{word}。', '她说你喜欢喝{word}，但我不确定。'),
}
HELD = {
    SPEAKER: ('我平时最爱喝{word}。', '我的最爱是不是{word}？', '告诉我我喜欢什么饮品。'),
    ADDRESSEE: ('你平时最爱喝{word}。', '你的最爱是不是{word}？', '告诉我你喜欢什么饮品。'),
    UNKNOWN: ('你和我各自喜欢什么饮料？', '朋友说：“我最爱喝{word}。”',
              '小雨喜欢的饮料是{word}。', '我喜欢{word}，你呢？'),
}
EXTERNAL_LABELS = {
    '0f74a3bb70e8f416432612e5bab35da02ca308cf3a5436237ce1fa89a5602f69': SPEAKER,
    '27e5f960ef2c4001df2961da1e3d748232cf140d03d7cf29060214397b52016b': SPEAKER,
    '8a88c02461ba56037a61b6e6e363ddd0c33a2ed46eb4a1fae052cee8bbe607b5': SPEAKER,
    '9b3b52b7ac2b2adc8cebd3d303fd1468ebeb08053871281c14825c93ab803624': ADDRESSEE,
    'd8e4ed13d7581737380b87bf1c3eb79814de04c9d70e860d130f29b36a1985eb': UNKNOWN,
    'e4b3a407fe603c8031beb99bf288717edca6b92d721c9875dedc0e751ffb2822': UNKNOWN,
    'fd9cde8df0a68ad48619595f82fafcc1377226d3a87e1c7bbacecb6f20686813': ADDRESSEE,
}


def resolve_relative(relative, role, self_key, participant_key, key_count):
    if (role not in ('user', 'assistant') or type(self_key) is not int or type(participant_key) is not int
            or type(key_count) is not int or key_count <= 0 or self_key == participant_key
            or not 0 <= self_key < key_count or not 0 <= participant_key < key_count
            or type(relative) is not int or relative not in (SPEAKER, ADDRESSEE, UNKNOWN)):
        raise ValueError('Invalid explicit participant binding')
    if relative == UNKNOWN:
        return None
    speaker = participant_key if role == 'user' else self_key
    listener = self_key if role == 'user' else participant_key
    return speaker if relative == SPEAKER else listener


def build_cases(templates):
    rows = []
    for label, choices in templates.items():
        for template in choices:
            words = WORDS if '{word}' in template else ('',)
            rows.extend({'text': template.format(word=word), 'label': label} for word in words)
    return rows


def evaluate(logits, rows):
    predicted = logits.argmax(-1).tolist()
    result = [dict(row, predicted=int(prediction), correct=int(prediction) == row['label'])
              for row, prediction in zip(rows, predicted)]
    return {'examples': len(result), 'correct': sum(r['correct'] for r in result), 'rows': result}


def run(output=OUTPUT, reader_path=None):
    output = Path(output).resolve()
    runs = (ROOT / 'runs').resolve()
    if not output.is_relative_to(runs) or output.exists():
        raise ValueError('Expected a new output directory under runs')
    reader_path = Path(reader_path).resolve() if reader_path is not None else None
    if reader_path is not None and not reader_path.is_relative_to(runs):
        raise ValueError('Reader must be under runs')
    files = {'weight': MODEL / 'model.safetensors', 'original': INDIVIDUAL, 'archive': ARCHIVE,
             'individual': CHECKPOINT / 'individual.pt', 'individual_report': CHECKPOINT / 'result.json',
             'value': VALUE / 'classifier.pt', 'value_report': VALUE / 'result.json'}
    if reader_path is not None:
        files.update(reader=reader_path, reader_report=reader_path.parent / 'result.json')
    before = {name: sha(path) for name, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['original'] != INDIVIDUAL_SHA or before['archive'] != ARCHIVE_SHA:
        raise ValueError('Protected source mismatch')
    with files['individual_report'].open(encoding='utf-8') as stream:
        checkpoint_report = json.load(stream)
    with files['value_report'].open(encoding='utf-8') as stream:
        value_report = json.load(stream)
    if checkpoint_report['checkpoint_sha256'] != before['individual'] or value_report['classifier_sha256'] != before['value']:
        raise ValueError('Component checkpoint mismatch')
    reader_data = None
    if reader_path is not None:
        with files['reader_report'].open(encoding='utf-8') as stream:
            reader_report = json.load(stream)
        reader_data = torch.load(reader_path, map_location='cpu', weights_only=True)
        if (reader_report['adapter_sha256'] != before['reader']
                or reader_data.get('format') != 'deepseek-spora-semantic-value-development-v1'
                or reader_data.get('weight_sha256') != WEIGHT_SHA or reader_data.get('individual_sha256') != INDIVIDUAL_SHA
                or reader_data.get('layers') != [5, 17] or reader_data.get('strength') != 0.5):
            raise ValueError('Reader provenance mismatch')
    config_sha = feature_files_sha256(MODEL)
    value_data = torch.load(files['value'], map_location='cpu', weights_only=True)
    if (value_data.get('format') != VALUE_FORMAT or value_data.get('weight_sha256') != WEIGHT_SHA
            or value_data.get('individual_sha256') != INDIVIDUAL_SHA
            or value_data.get('feature_config_sha256') != config_sha or value_data.get('hidden_indices') != [6, 18]
            or tuple(value_data.get('vocabulary', ())) != WORDS or value_data.get('abstain') != ABSTAIN):
        raise ValueError('Value component provenance mismatch')
    train, held = build_cases(TRAIN), build_cases(HELD)
    external = sample_archive(ARCHIVE, topic=True)
    if set(r['text_sha256'] for r in external) != set(EXTERNAL_LABELS):
        raise ValueError('External audit set differs from annotated sample')
    external = [dict(row, label=EXTERNAL_LABELS[row['text_sha256']]) for row in external]
    if {r['text'] for r in train} & ({r['text'] for r in held} | {r['text'] for r in external}):
        raise ValueError('Training and audit text overlap')
    events = [
        {'role': 'user', 'participant_key': 22, 'text': '我喜欢喝水。', 'expected_key': 22, 'expected_value': 6},
        {'role': 'assistant', 'participant_key': 22, 'text': '我喜欢喝咖啡。', 'expected_key': 6, 'expected_value': 1},
        {'role': 'user', 'participant_key': 22, 'text': '我最喜欢的饮料是茶。', 'expected_key': 22, 'expected_value': 0},
        {'role': 'user', 'participant_key': 22, 'text': '你不喜欢喝茶。', 'expected_key': None, 'expected_value': None},
        {'role': 'user', 'participant_key': 35, 'text': '我最喜欢的饮料是豆浆。', 'expected_key': 35, 'expected_value': 4},
    ]
    questions = [
        {'role': 'user', 'participant_key': 22, 'text': '我喜欢的饮品是什么？', 'expected_key': 22, 'expected_value': 0},
        {'role': 'user', 'participant_key': 22, 'text': '你喜欢的饮品是什么？', 'expected_key': 6, 'expected_value': 1},
        {'role': 'assistant', 'participant_key': 22, 'text': '我喜欢的饮品是什么？', 'expected_key': 6, 'expected_value': 1},
        {'role': 'assistant', 'participant_key': 22, 'text': '你喜欢的饮品是什么？', 'expected_key': 22, 'expected_value': 0},
        {'role': 'user', 'participant_key': 35, 'text': '我喜欢的饮品是什么？', 'expected_key': 35, 'expected_value': 4},
        {'role': 'user', 'participant_key': 35, 'text': '你喜欢的饮品是什么？', 'expected_key': 6, 'expected_value': 1},
    ]
    torch.manual_seed(4643)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, saved = load_integrated(files['individual'], base, MODEL)
    if reader_data is not None:
        integrated.reader.load_state_dict(reader_data['state'], strict=True)
    integrated.eval()
    saved_digest = frozen_digest(organism, saved)
    splits = {'train': train, 'held': held, 'external': external, 'events': events, 'questions': questions}
    vectors = {}
    for name, rows in splits.items():
        features, masks = extract(base, tokenizer, [r['text'] for r in rows], batch_size=2)
        vectors[name] = pool(features, masks)
        print(f'{name} features ready: {len(rows)}', flush=True)
    subject = nn.Linear(vectors['train'].shape[-1], 3)
    labels = torch.tensor([r['label'] for r in train])
    optimizer = torch.optim.AdamW(subject.parameters(), lr=0.0002, weight_decay=0.01)
    for _ in range(600):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(subject(vectors['train']), labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite subject loss')
        loss.backward()
        optimizer.step()
    subject.eval()
    value = FactorizedProbe(vectors['events'].shape[-1]).eval()
    value.load_state_dict(value_data['state'], strict=True)
    with torch.no_grad():
        results = {name: evaluate(subject(vectors[name]), rows) for name, rows in
                   (('train', train), ('held', held), ('external', external))}
        event_subject = subject(vectors['events']).argmax(-1).tolist()
        question_subject = subject(vectors['questions']).argmax(-1).tolist()
        fact, values = value(vectors['events'])
        event_values = combined_logits(fact, values).argmax(-1).tolist()
    state = fresh_controlled_state(organism)
    event_results = []
    for row, relative, predicted_value in zip(events, event_subject, event_values):
        key = resolve_relative(int(relative), row['role'], 6, row['participant_key'], organism.config.key_count)
        decision = None
        if key is not None and predicted_value != ABSTAIN:
            state, decision = observe_sourced(organism, state, key, int(predicted_value), 'direct', 1.0)
        event_results.append({**row, 'relative': relative, 'predicted_key': key, 'predicted_value': predicted_value,
                              'decision': decision, 'correct_write': row['expected_key'] is not None
                              and key == row['expected_key'] and predicted_value == row['expected_value']
                              and decision is not None and decision['current_write'],
                              'unexpected_write': row['expected_key'] is None and decision is not None
                              and decision['current_write']})
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    question_results = []
    for row, relative in zip(questions, question_subject):
        key = resolve_relative(int(relative), row['role'], 6, row['participant_key'], organism.config.key_count)
        status, answer = None, None
        if key is not None:
            status = current_memory_status(organism, state, key)['status']
            if status == 'known':
                inputs = closed_think_inputs(tokenizer, row['text'], 'cuda')
                addresses = [basis[key:key + 1] for basis in bases]
                with torch.no_grad():
                    generated = integrated.generate(**inputs, memory_state=state, memory_addresses=addresses,
                                                     current_only=True, max_new_tokens=8, do_sample=False,
                                                     pad_token_id=tokenizer.eos_token_id)
                answer = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        question_results.append({**row, 'relative': relative, 'predicted_key': key, 'status': status, 'answer': answer,
                                 'correct': key == row['expected_key'] and answer is not None
                                 and word_correct(answer, WORDS[row['expected_value']])})
    keys = torch.tensor([6, 22, 35], dtype=torch.long)
    with torch.inference_mode():
        native = query(organism, state, keys, torch.zeros_like(keys), torch.zeros_like(keys)).argmax(-1).tolist()
    if (any(sha(path) != before[name] for name, path in files.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, saved) != saved_digest or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected input changed')
    output.mkdir(parents=False, exist_ok=False)
    checkpoint = output / 'subject.pt'
    torch.save({'format': 'deepseek-relative-subject-development-v1', 'state': subject.state_dict(),
                'weight_sha256': WEIGHT_SHA, 'feature_config_sha256': config_sha, 'hidden_indices': [6, 18]}, checkpoint)
    report = {'status': 'relative_identity_isolated_research_only', 'subject_sha256': sha(checkpoint),
              'results': results, 'events': event_results, 'questions': question_results,
              'native_values_self_then_participants': native, 'component_sha256': before,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Self and two participants use manually assigned experimental slots; no identity authentication or slot management.',
                              'Both users and assistants are simulated as direct trusted sources; assistant self-claims are not automatically trustworthy.',
                              'The value extractor and readout remain fixed-vocabulary experiments; input templates are synthetic.',
                              'Questions are explicitly sent to the read path; no general dialogue router or actual user writes.',
                              'Seven external labels are manually assigned to isolated single utterances; context and pretraining overlap remain unknown.',
                              'Fresh temporary state is not persisted and does not replace the original individual.']}
    with (output / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'subject_held': results['held']['correct'], 'subject_held_total': results['held']['examples'],
                      'subject_external': results['external']['correct'], 'subject_external_total': results['external']['examples'],
                      'correct_writes': sum(r['correct_write'] for r in event_results),
                      'unexpected_writes': sum(r['unexpected_write'] for r in event_results),
                      'correct_answers': sum(r['correct'] for r in question_results)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--reader', type=Path)
    args = parser.parse_args()
    run(args.output, args.reader)
