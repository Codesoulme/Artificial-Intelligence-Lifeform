"""Evaluate two independently trained local writer corrections together."""
import argparse
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
from scripts.deepseek_spora_writer_correction_pilot import OUT as RED, SOURCE
from scripts.deepseek_spora_writer_layer_control import collect
from spora.language_training import atomic_save
from spora.research_local_writer import FactorizedLocalWriterBank, LocalWriterBank, calibrate_threshold
from spora.research_semantic_slot import install_sequence_scaled, use_slot_sequence
from spora.semantic_text_writer import value_positions

SOY = ROOT / 'runs/deepseek-spora-writer-correction-soy-20261116'
GATE = ROOT / 'runs/deepseek-spora-local-writer-gate-20261116.pt'
GATE_REPORT = ROOT / 'runs/deepseek-spora-local-writer-artifact-audit-20261116.json'
PREVIOUS = ROOT / 'runs/deepseek-spora-local-writer-audit-20261116.json'
OUT = ROOT / 'runs/deepseek-spora-local-writer-bank-20261116.json'
ARTIFACT = ROOT / 'runs/deepseek-spora-local-writer-bank-20261116.pt'


def main(output=OUT, artifact=ARTIFACT, candidate_input=None, expected_candidate_sha=None,
         factorized=False, factor_output=None, factor_input=None, expected_factor_sha=None):
    if output.exists() or (candidate_input is None and artifact.exists()):
        raise FileExistsError('Audit output or candidate already exists')
    if candidate_input is not None and not expected_candidate_sha:
        raise ValueError('Reload requires a candidate digest')
    if factorized and candidate_input is None:
        raise ValueError('Factorized comparison requires an existing candidate')
    if factor_output is not None and (not factorized or factor_output.exists()):
        raise ValueError('Factor output requires factorized mode and a new file')
    if factor_input is not None and (not factorized or factor_output is not None or not expected_factor_sha):
        raise ValueError('Factor reload requires factorized mode, a digest, and no factor output')
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    original_report = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    reports = [json.loads((path / 'result.json').read_text(encoding='utf-8')) for path in (RED, SOY)]
    gate_report = json.loads(GATE_REPORT.read_text(encoding='utf-8'))
    old_result = json.loads(PREVIOUS.read_text(encoding='utf-8'))
    writer_paths = [SOURCE / 'writer.pt', RED / 'writer.pt', SOY / 'writer.pt']
    hashes = [base.sha(path) for path in writer_paths]
    if (original_report['writer_sha256'] != hashes[0] or any(
            report['writer_sha256'] != sha or report['source_writer_sha256'] != hashes[0]
            or report['base_sha256'] != base.WEIGHT_SHA for report, sha in zip(reports, hashes[1:]))
            or base.sha(GATE) != gate_report['candidate_sha256']):
        raise ValueError('Writer or gate provenance mismatch')
    original, red, soy = [torch.load(path, map_location='cpu', weights_only=True) for path in writer_paths]
    gate = torch.load(GATE, map_location='cpu', weights_only=True)
    if (gate['base_sha256'] != base.WEIGHT_SHA or gate['original_writer_sha256'] != hashes[0]
            or gate['corrected_writer_sha256'] != hashes[1] or gate['gain'] != GAIN
            or gate['train_norm'] != original['train_norm']):
        raise ValueError('Prior gate binding mismatch')
    if reports[1]['subject'] != '小赵' or reports[1]['target'] != '豆浆':
        raise ValueError('Unexpected second correction')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    layers = {i: model.model.layers[i] for i in (6, 12)}
    paths = {i: PAIR / f'layer-{i}.pt' for i in layers}
    layer_hashes = {str(i): base.sha(path) for i, path in paths.items()}
    if any(report['layers'] != layer_hashes for report in reports) or gate['replacement_sha256'] != layer_hashes:
        raise ValueError('Replacement sources changed')
    replacements = {i: _load_candidate(path, i, model).eval() for i, path in paths.items()}
    for i in layers:
        model.model.layers[i] = replacements[i]
    if candidate_input is None:
        cases = [{'name': reports[1]['subject'], 'value': reports[1]['target']}]
        cases += [{'name': person, 'value': value} for person in prior.TRAIN_NAMES
                  for value in reports[1]['replay_values']]
        features, _ = collect(model, tokenizer, cases)
        values = features['shallow_layer_1']
        soy_anchor = values[0]
        soy_threshold = calibrate_threshold(soy_anchor, torch.cat((values[1:], gate['anchor'].unsqueeze(0))))
        anchors, thresholds = [gate['anchor'], soy_anchor], [gate['threshold'], soy_threshold]
        candidate_sha = None
    else:
        candidate_sha = base.sha(candidate_input)
        if candidate_sha != expected_candidate_sha:
            raise ValueError('Candidate digest mismatch')
        stored = torch.load(candidate_input, map_location='cpu', weights_only=True)
        if (stored.get('format') != 'spora-local-writer-bank-v1' or stored.get('base_sha256') != base.WEIGHT_SHA
                or stored.get('writer_sha256') != hashes or stored.get('layer_sha256') != layer_hashes
                or stored.get('gain') != GAIN or stored.get('train_norm') != original['train_norm']
                or stored['anchors'].shape != (2, model.config.hidden_size) or stored['thresholds'].shape != (2,)):
            raise ValueError('Candidate binding mismatch')
        anchors = list(stored['anchors'])
        thresholds = stored['thresholds'].tolist()
        soy_threshold = thresholds[1]
    writer = LocalWriterBank(original['state']['projection.weight'],
                             [red['state']['projection.weight'], soy['state']['projection.weight']],
                             anchors, thresholds).to('cuda').eval()
    factorization = None
    loaded_factor_sha = None
    if factorized:
        downs, ups, errors = [], [], []
        if factor_input is not None:
            loaded_factor_sha = base.sha(factor_input)
            if loaded_factor_sha != expected_factor_sha:
                raise ValueError('Factor candidate digest mismatch')
            factor_payload = torch.load(factor_input, map_location='cpu', weights_only=True)
            if (factor_payload.get('format') != 'spora-local-writer-factors-v1'
                    or factor_payload.get('base_sha256') != base.WEIGHT_SHA
                    or factor_payload.get('writer_sha256') != hashes
                    or factor_payload.get('layer_sha256') != layer_hashes
                    or factor_payload.get('dense_candidate_sha256') != candidate_sha
                    or factor_payload.get('gain') != GAIN
                    or factor_payload.get('train_norm') != original['train_norm']
                    or not torch.allclose(factor_payload['anchors'], writer.anchors.cpu(), rtol=0., atol=1e-7)
                    or not torch.equal(factor_payload['thresholds'], writer.thresholds.cpu())):
                raise ValueError('Factor candidate binding mismatch')
            downs = list(factor_payload['down'].to('cuda'))
            ups = list(factor_payload['up'].to('cuda'))
        else:
            torch.manual_seed(81317)
            for delta in writer.delta_weights:
                u, singular, v = torch.svd_lowrank(delta, q=16, niter=4)
                ups.append(u[:, :8] * singular[:8])
                downs.append(v[:, :8].T.contiguous())
        if len(downs) != len(writer.delta_weights) or len(ups) != len(writer.delta_weights):
            raise ValueError('Factor count mismatch')
        for delta, up, down in zip(writer.delta_weights, ups, downs):
            difference = delta - up @ down
            errors.append({'max_abs': float(difference.abs().max()),
                           'relative_frobenius': float(difference.norm() / delta.norm())})
        writer = FactorizedLocalWriterBank(writer.original_weight, torch.stack(downs), torch.stack(ups),
                                           writer.anchors, writer.thresholds).to('cuda').eval()
        factorization = {'rank': 8, 'errors': errors,
                         'dense_delta_bytes': len(downs) * writer.original_weight.numel() * 4,
                         'factor_delta_bytes': (writer.down.numel() + writer.up.numel()) * 4}
    install_sequence_scaled(model, scale=GAIN)

    def generate(name, value, question):
        encoded, positions = value_positions(tokenizer, f'{name}喜欢喝{value}。', value)
        encoded = {k: v.to('cuda') for k, v in encoded.items()}
        target_ids = encoded['input_ids'][0, positions].tolist()
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        with torch.inference_mode():
            hidden = model.model(**encoded, use_cache=False, output_hidden_states=True,
                                 return_dict=True).hidden_states[1][0, positions].float()
            matches = writer.matches(hidden).tolist()
            slots = norm_vectors(writer(hidden), original['train_norm'])
            eos = model.model.embed_tokens.weight[tokenizer.eos_token_id].float().unsqueeze(0)
            with use_slot_sequence(torch.cat((slots, eos), dim=0)):
                result = model.generate(**prompt, use_cache=True, do_sample=False,
                                        max_new_tokens=len(target_ids) + 3,
                                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tail = result[0, prompt['input_ids'].shape[1]:].tolist()
        return {'ids': tail, 'exact': tail == target_ids + [tokenizer.eos_token_id], 'matches': matches}

    phrase_rows = []
    for name in prior.HELD_NAMES:
        for value in VALUES:
            for template in QUESTIONS:
                question = template.format(name=name)
                phrase_rows.append({'name': name, 'value': value, 'question': question,
                                    'result': generate(name, value, question)})
        print(json.dumps({'phrase_name_done': name}, ensure_ascii=True), flush=True)
    old_phrases = old_result['phrase_rows']
    if [(r['name'], r['value'], r['question']) for r in phrase_rows] != [
            (r['name'], r['value'], r['question']) for r in old_phrases]:
        raise ValueError('Previous phrase cases do not match')
    for i in layers:
        model.model.layers[i] = layers[i]
    token_rows = []
    for name in prior.HELD_NAMES:
        for value in original_report['held_values']:
            token_rows.append({'name': name, 'value': value,
                               'result': generate(name, value, f'{name}喜欢喝什么饮料？')})
        print(json.dumps({'token_name_done': name}, ensure_ascii=True), flush=True)
    if [(r['name'], r['value']) for r in token_rows] != [
            (r['name'], r['value']) for r in old_result['token_rows']]:
        raise ValueError('Previous token cases do not match')
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or any(base.sha(path) != sha for path, sha in zip(writer_paths, hashes))
            or base.sha(GATE) != gate_report['candidate_sha256']
            or (candidate_input is not None and base.sha(candidate_input) != candidate_sha)
            or (factor_input is not None and base.sha(factor_input) != loaded_factor_sha)
            or any(base.sha(path) != layer_hashes[str(i)] for i, path in paths.items())):
        raise ValueError('Protected source changed')
    targets = {word: [row for row in phrase_rows if row['value'] == word] for word in ('红茶', '豆浆')}
    all_rows = phrase_rows + token_rows
    summary = {'phrase_exact': sum(row['result']['exact'] for row in phrase_rows),
               'previous_phrase_exact': sum(row['result']['exact'] for row in old_phrases),
               'target_exact': {word: sum(row['result']['exact'] for row in group) for word, group in targets.items()},
               'previous_target_exact': {word: sum(row['result']['exact'] for row in old_phrases
                                                     if row['value'] == word) for word in targets},
               'held_token_exact': sum(row['result']['exact'] for row in token_rows),
               'previous_held_token_exact': sum(row['result']['exact'] for row in old_result['token_rows']),
               'held_token_unchanged': sum(row['result']['ids'] == old['result']['ids']
                                           for row, old in zip(token_rows, old_result['token_rows'])),
               'ambiguous_token_count': sum(sum(matches) > 1 for row in all_rows
                                             for matches in row['result']['matches']),
               'single_gate_token_counts': [sum(matches[i] and sum(matches) == 1 for row in all_rows
                                                for matches in row['result']['matches']) for i in (0, 1)]}
    if candidate_input is None:
        atomic_save({'format': 'spora-local-writer-bank-v1', 'base_sha256': base.WEIGHT_SHA,
                     'writer_sha256': hashes, 'layer_sha256': layer_hashes, 'gain': GAIN,
                     'train_norm': original['train_norm'], 'anchors': writer.anchors.detach().cpu(),
                     'thresholds': writer.thresholds.detach().cpu()}, artifact)
        candidate_sha = base.sha(artifact)
    factor_sha = loaded_factor_sha
    if factor_output is not None:
        atomic_save({'format': 'spora-local-writer-factors-v1', 'base_sha256': base.WEIGHT_SHA,
                     'writer_sha256': hashes, 'layer_sha256': layer_hashes, 'gain': GAIN,
                     'train_norm': original['train_norm'], 'anchors': writer.anchors.detach().cpu(),
                     'thresholds': writer.thresholds.detach().cpu(), 'down': writer.down.detach().cpu(),
                     'up': writer.up.detach().cpu(), 'dense_candidate_sha256': candidate_sha}, factor_output)
        factor_sha = base.sha(factor_output)
    report = {'summary': summary, 'soy_threshold': soy_threshold, 'red_threshold': gate['threshold'],
              'artifact_sha256': candidate_sha, 'phrase_rows': phrase_rows, 'token_rows': token_rows,
              'factorization': factorization, 'factor_sha256': factor_sha,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two synthetic target words with manually supplied spans and fixed routing.',
                              'The second gate uses one positive and 128 replay negatives plus the first anchor.',
                              'Both corrections start from the same writer; this is not online sequential training.',
                              'Ambiguous matches fall back to base and are not automatically resolved.',
                              'Artificial EOS, no autonomous learning or persistent personal memory.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--artifact', type=Path, default=ARTIFACT)
    parser.add_argument('--candidate-input', type=Path)
    parser.add_argument('--expected-candidate-sha')
    parser.add_argument('--factorized', action='store_true')
    parser.add_argument('--factor-output', type=Path)
    parser.add_argument('--factor-input', type=Path)
    parser.add_argument('--expected-factor-sha')
    args = parser.parse_args()
    main(args.output, args.artifact, args.candidate_input, args.expected_candidate_sha,
         args.factorized, args.factor_output, args.factor_input, args.expected_factor_sha)
