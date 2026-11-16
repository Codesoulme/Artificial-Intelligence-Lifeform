"""Stress a single-anchor writer gate on altered names, phrases and values."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch.nn import functional as F
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_text_writer_broad_generalization as prior
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from scripts.deepseek_spora_output_aware_writer import GAIN, norm_vectors
from scripts.deepseek_spora_writer_correction_pilot import OUT as CORRECTED, SOURCE, SUBJECT, TARGET
from scripts.deepseek_spora_writer_layer_control import collect
from spora.language_training import atomic_save
from spora.research_local_writer import LocalWriter, calibrate_threshold
from spora.research_semantic_slot import install_sequence_scaled, use_slot_sequence
from spora.semantic_text_writer import value_positions

OUT = ROOT / 'runs/deepseek-spora-local-writer-stress-20261116.json'
NAMES = ('小郑', '小赵', '林婉清', '张子瑜', '李明', '陈晓', '王宁', '阿岚')
TEMPLATES = ('{name}喜欢喝{value}。', '{name}最喜欢喝{value}。',
             '听说{name}常常选择{value}。', '{name}说过自己很喜欢{value}。')
OTHER_VALUES = ('绿茶', '咖啡', '豆浆')


def main(output=OUT, generate=False, candidate=None, candidate_input=None, expected_candidate_sha=None):
    if output.exists():
        raise FileExistsError(output)
    if candidate is not None and candidate.exists():
        raise FileExistsError(candidate)
    if candidate_input is not None and (candidate is not None or not expected_candidate_sha):
        raise ValueError('Reload requires a digest and cannot create a candidate')
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    report = json.loads((CORRECTED / 'result.json').read_text(encoding='utf-8'))
    source = CORRECTED / 'writer.pt'
    if report['writer_sha256'] != base.sha(source) or report['base_sha256'] != base.WEIGHT_SHA:
        raise ValueError('Correction provenance mismatch')
    paths = {i: PAIR / f'layer-{i}.pt' for i in (6, 12)}
    if any(base.sha(path) != report['layers'][str(i)] for i, path in paths.items()):
        raise ValueError('Replacement provenance mismatch')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed(skip_replaced=(6, 12)).eval()
    for index, path in paths.items():
        model.model.layers[index] = _load_candidate(path, index, model).eval()
    original_sha = base.sha(SOURCE / 'writer.pt')
    if original_sha != report['source_writer_sha256']:
        raise ValueError('Original writer provenance mismatch')
    original = torch.load(SOURCE / 'writer.pt', map_location='cpu', weights_only=True)
    corrected = torch.load(source, map_location='cpu', weights_only=True)
    loaded_sha = None
    if candidate_input is None:
        cases = [{'name': SUBJECT, 'value': TARGET}]
        cases += [{'name': person, 'value': value} for person in prior.TRAIN_NAMES
                  for value in report['replay_values']]
        features, _ = collect(model, tokenizer, cases)
        vectors = features['shallow_layer_1']
        anchor = F.normalize(vectors[0].to('cuda'), dim=0)
        threshold = calibrate_threshold(vectors[0], vectors[1:])
    else:
        loaded_sha = base.sha(candidate_input)
        if loaded_sha != expected_candidate_sha:
            raise ValueError('Gate candidate digest mismatch')
        stored = torch.load(candidate_input, map_location='cpu', weights_only=True)
        if (stored.get('format') != 'spora-local-writer-gate-v1'
                or stored.get('base_sha256') != base.WEIGHT_SHA
                or stored.get('original_writer_sha256') != original_sha
                or stored.get('corrected_writer_sha256') != report['writer_sha256']
                or stored.get('replacement_sha256') != report['layers']
                or stored.get('gain') != GAIN or stored.get('train_norm') != original['train_norm']):
            raise ValueError('Gate candidate binding mismatch')
        anchor = stored['anchor'].to('cuda')
        threshold = stored['threshold']
    writer = LocalWriter(original['state']['projection.weight'], corrected['state']['projection.weight'],
                         anchor, threshold).to('cuda').eval()
    if generate:
        install_sequence_scaled(model, scale=GAIN)
    red_ids = tokenizer.encode(TARGET, add_special_tokens=False)
    if len(red_ids) != 1:
        raise ValueError('Expected one target token')
    rows = []
    for name in NAMES:
        for value, templates in ((TARGET, TEMPLATES), *((value, TEMPLATES[:1]) for value in OTHER_VALUES)):
            for template in templates:
                text = template.format(name=name, value=value)
                try:
                    encoded, positions = value_positions(tokenizer, text, value)
                except ValueError:
                    rows.append({'name': name, 'value': value, 'template': template, 'status': 'unaligned'})
                    continue
                encoded = {k: v.to('cuda') for k, v in encoded.items()}
                ids = encoded['input_ids'][0, positions].tolist()
                with torch.inference_mode():
                    hidden = model.model(**encoded, use_cache=False, output_hidden_states=True,
                                         return_dict=True).hidden_states[1][0, positions].float()
                    scores = (F.normalize(hidden, dim=-1) * anchor).sum(-1).tolist()
                    outputs = None
                    if generate:
                        question = pilot.closed_think_inputs(tokenizer, f'{name}喜欢喝什么饮料？', 'cuda')
                        base_vectors = F.linear(hidden, writer.original_weight)
                        eos = model.model.embed_tokens.weight[tokenizer.eos_token_id].float().unsqueeze(0)
                        outputs = {}
                        for mode, predicted in (('original', base_vectors), ('local', writer(hidden))):
                            slots = torch.cat((norm_vectors(predicted, original['train_norm']), eos), dim=0)
                            with use_slot_sequence(slots):
                                result = model.generate(**question, use_cache=True, do_sample=False,
                                                        max_new_tokens=len(ids) + 3,
                                                        pad_token_id=tokenizer.pad_token_id,
                                                        eos_token_id=tokenizer.eos_token_id)
                            tail = result[0, question['input_ids'].shape[1]:].tolist()
                            outputs[mode] = {'ids': tail, 'exact': tail == ids + [tokenizer.eos_token_id]}
                rows.append({'name': name, 'value': value, 'template': template, 'status': 'ok',
                             'ids': ids, 'scores': scores, 'active': [score > threshold for score in scores],
                             'outputs': outputs})
        print(json.dumps({'completed_name': name}, ensure_ascii=True), flush=True)
    if (base.sha(source) != report['writer_sha256'] or base.sha(SOURCE / 'writer.pt') != original_sha
            or base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or (candidate_input is not None and base.sha(candidate_input) != loaded_sha)
            or any(base.sha(path) != report['layers'][str(i)] for i, path in paths.items())):
        raise ValueError('Protected input changed')
    red = [r for r in rows if r['value'] == TARGET and r['status'] == 'ok']
    other = [r for r in rows if r['value'] != TARGET and r['status'] == 'ok']
    summary = {'target_aligned': len(red), 'target_active': sum(any(r['active']) for r in red),
               'other_aligned': len(other), 'other_active': sum(any(r['active']) for r in other),
               'unaligned': sum(r['status'] != 'ok' for r in rows), 'cases': len(rows)}
    if generate:
        for label, subset in (('target', red), ('other', other)):
            for mode in ('original', 'local'):
                summary[f'{label}_{mode}_exact'] = sum(r['outputs'][mode]['exact'] for r in subset)
    candidate_sha = None
    if candidate is not None:
        atomic_save({'format': 'spora-local-writer-gate-v1', 'anchor': writer.anchor.detach().cpu(),
                     'threshold': threshold, 'base_sha256': base.WEIGHT_SHA,
                     'original_writer_sha256': original_sha, 'corrected_writer_sha256': report['writer_sha256'],
                     'replacement_sha256': report['layers'], 'gain': GAIN, 'train_norm': original['train_norm']},
                    candidate)
        candidate_sha = base.sha(candidate)
    result = {'threshold': threshold, 'summary': summary, 'rows': rows, 'base_sha256': base.WEIGHT_SHA,
              'writer_sha256': report['writer_sha256'], 'candidate_sha256': candidate_sha,
              'loaded_candidate_sha256': loaded_sha,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Optional oracle-slot generation; no memory writes or natural fact routing.',
                              'Oracle value boundaries, one target word, artificial names and sentences.',
                              'Unaligned token spans are reported separately and not counted as safe rejections.',
                              'Existing project material and model pretraining may overlap these examples.',
                              'Different factuality or negation is not interpreted by this gate.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--candidate-input', type=Path)
    parser.add_argument('--expected-candidate-sha')
    args = parser.parse_args()
    main(args.output, args.generate, args.candidate, args.candidate_input, args.expected_candidate_sha)
