"""Audit a feature-gated correction against its original and global controls."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_text_writer_broad_generalization as prior
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_broad_writer_phrases import QUESTIONS, VALUES
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from scripts.deepseek_spora_output_aware_writer import GAIN, norm_vectors
from scripts.deepseek_spora_writer_correction_pilot import SOURCE, OUT as CORRECTED, SUBJECT, TARGET
from scripts.deepseek_spora_writer_layer_control import collect
from spora.research_local_writer import LocalWriter, calibrate_threshold
from spora.research_semantic_slot import install_sequence_scaled, use_slot_sequence
from spora.semantic_text_writer import value_positions

OUT = ROOT / 'runs/deepseek-spora-local-writer-audit-20261116.json'
PHRASE_BASE = ROOT / 'runs/deepseek-spora-broad-writer-phrases-replaced-20261116.json'
PHRASE_GLOBAL = ROOT / 'runs/deepseek-spora-writer-correction-phrases-20261116.json'
TOKEN_BASE = ROOT / 'runs/deepseek-spora-output-aware-broad-reload-20261116.json'
TOKEN_GLOBAL = ROOT / 'runs/deepseek-spora-writer-correction-reload-20261116.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    source_report = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    correction_report = json.loads((CORRECTED / 'result.json').read_text(encoding='utf-8'))
    source_path, corrected_path = SOURCE / 'writer.pt', CORRECTED / 'writer.pt'
    protected = {str(path): base.sha(path) for path in
                 (source_path, corrected_path, PHRASE_BASE, PHRASE_GLOBAL, TOKEN_BASE, TOKEN_GLOBAL)}
    if (protected[str(source_path)] != source_report['writer_sha256']
            or protected[str(corrected_path)] != correction_report['writer_sha256']
            or correction_report['source_writer_sha256'] != protected[str(source_path)]):
        raise ValueError('Writer provenance mismatch')
    original = torch.load(source_path, map_location='cpu', weights_only=True)
    corrected = torch.load(corrected_path, map_location='cpu', weights_only=True)
    if original['base_sha256'] != base.WEIGHT_SHA or corrected['base_sha256'] != base.WEIGHT_SHA:
        raise ValueError('Writer base mismatch')
    phrase_base = json.loads(PHRASE_BASE.read_text(encoding='utf-8'))['rows']
    phrase_global = json.loads(PHRASE_GLOBAL.read_text(encoding='utf-8'))['rows']
    token_base = json.loads(TOKEN_BASE.read_text(encoding='utf-8'))['rows']
    token_global = json.loads(TOKEN_GLOBAL.read_text(encoding='utf-8'))['rows']
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    layers = {i: model.model.layers[i] for i in (6, 12)}
    replacement_paths = {i: PAIR / f'layer-{i}.pt' for i in layers}
    replacement_sha = {str(i): base.sha(path) for i, path in replacement_paths.items()}
    if replacement_sha != correction_report['layers']:
        raise ValueError('Correction replacement sources changed')
    replacements = {i: _load_candidate(path, i, model).eval() for i, path in replacement_paths.items()}
    for i in layers:
        model.model.layers[i] = replacements[i]
    cases = [{'name': SUBJECT, 'value': TARGET}]
    cases += [{'name': person, 'value': value} for person in prior.TRAIN_NAMES
              for value in correction_report['replay_values']]
    features, _ = collect(model, tokenizer, cases)
    vectors = features['shallow_layer_1']
    threshold = calibrate_threshold(vectors[0], vectors[1:])
    writer = LocalWriter(original['state']['projection.weight'], corrected['state']['projection.weight'],
                         vectors[0], threshold).to('cuda').eval()
    with torch.inference_mode():
        calibration_positive = bool(writer.active(vectors[:1].to('cuda'))[0])
        calibration_false = int(writer.active(vectors[1:].to('cuda')).sum())
    install_sequence_scaled(model, scale=GAIN)

    def get_result(name, value, question):
        encoded, positions = value_positions(tokenizer, f'{name}喜欢喝{value}。', value)
        encoded = {k: v.to('cuda') for k, v in encoded.items()}
        target_ids = encoded['input_ids'][0, positions].tolist()
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        with torch.inference_mode():
            hidden = model.model(**encoded, use_cache=False, output_hidden_states=True,
                                 return_dict=True).hidden_states[1][0, positions].float()
            active = writer.active(hidden).tolist()
            slots = norm_vectors(writer(hidden), original['train_norm'])
            eos = model.model.embed_tokens.weight[tokenizer.eos_token_id].float().unsqueeze(0)
            with use_slot_sequence(torch.cat((slots, eos), dim=0)):
                result = model.generate(**prompt, use_cache=True, do_sample=False,
                                        max_new_tokens=len(target_ids) + 3,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id)
            tail = result[0, prompt['input_ids'].shape[1]:].tolist()
        return {'ids': tail, 'text': tokenizer.decode(tail, skip_special_tokens=True),
                'exact': tail == target_ids + [tokenizer.eos_token_id], 'active': active}

    phrase_rows = []
    for name in prior.HELD_NAMES:
        for value in VALUES:
            for template in QUESTIONS:
                question = template.format(name=name)
                phrase_rows.append({'name': name, 'value': value, 'question': question,
                                    'result': get_result(name, value, question)})
        print(json.dumps({'phrase_name_done': name}), flush=True)
    if ([(r['name'], r['value'], r['question']) for r in phrase_rows]
            != [(r['name'], r['value'], r['question']) for r in phrase_base]
            or [(r['name'], r['value'], r['question']) for r in phrase_rows]
            != [(r['name'], r['value'], r['question']) for r in phrase_global]):
        raise ValueError('Phrase controls do not match')
    for i in layers:
        model.model.layers[i] = layers[i]
    token_rows = []
    for name in prior.HELD_NAMES:
        for value in source_report['held_values']:
            question = f'{name}喜欢喝什么饮料？'
            token_rows.append({'name': name, 'value': value, 'result': get_result(name, value, question)})
        print(json.dumps({'token_name_done': name}), flush=True)
    token_base = [r for r in token_base if r['name_split'] == 'held' and r['value_split'] == 'held']
    token_global = [r for r in token_global if r['name_split'] == 'held' and r['value_split'] == 'held']
    if ([(r['name'], r['value']) for r in token_rows] != [(r['name'], r['value']) for r in token_base]
            or [(r['name'], r['value']) for r in token_rows] != [(r['name'], r['value']) for r in token_global]):
        raise ValueError('Token controls do not match')
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or any(base.sha(Path(path)) != sha for path, sha in protected.items())
            or any(base.sha(path) != replacement_sha[str(i)] for i, path in replacement_paths.items())):
        raise ValueError('Protected source changed')
    red = [r for r in phrase_rows if r['value'] == TARGET]
    nonred = [r for r in phrase_rows if r['value'] != TARGET]
    summary = {'calibration_positive_active': calibration_positive, 'calibration_negative_active': calibration_false,
               'calibration_negative_count': len(cases) - 1,
               'phrase_exact': sum(r['result']['exact'] for r in phrase_rows),
               'target_exact': sum(r['result']['exact'] for r in red),
               'target_active': sum(any(r['result']['active']) for r in red),
               'other_phrase_active': sum(any(r['result']['active']) for r in nonred),
               'held_token_exact': sum(r['result']['exact'] for r in token_rows),
               'held_token_active': sum(any(r['result']['active']) for r in token_rows),
               'held_token_matches_base': sum(r['result']['ids'] == b['generated_ids']
                                              for r, b in zip(token_rows, token_base)),
               'held_token_matches_global': sum(r['result']['ids'] == b['generated_ids']
                                                for r, b in zip(token_rows, token_global))}
    report = {'summary': summary, 'threshold': threshold, 'phrase_rows': phrase_rows, 'token_rows': token_rows,
              'source_sha256': protected, 'replacement_sha256': replacement_sha,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Threshold calibrated on one positive and 128 synthetic replay negatives.',
                              'The correction target and examples were already used in prior development.',
                              'Oracle span, identity and EOS; no natural routing or persistent individual.',
                              'Substring similarity is not proof of semantic identity or authorization.',
                              'Phrase and token samples share names and values, not independent observations.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
