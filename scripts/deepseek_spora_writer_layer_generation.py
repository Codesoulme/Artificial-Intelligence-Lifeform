"""Check layer-control writers through the existing oracle sequence bridge."""
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
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_semantic_slot import install_sequence_scaled, use_slot_sequence
from spora.semantic_text_writer import TextSemanticWriter, value_positions

SOURCE = ROOT / 'runs/deepseek-spora-writer-identity-control-20261116'
OUT = ROOT / 'runs/deepseek-spora-writer-layer-generation-20261116.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    source_report = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    source_path = SOURCE / 'writers.pt'
    source_sha = base.sha(source_path)
    if (source_report['base_sha256'] != base.WEIGHT_SHA or source_report['weights_sha256'] != source_sha
            or not source_report['identity_initialization']):
        raise ValueError('Writer provenance mismatch')
    payload = torch.load(source_path, map_location='cpu', weights_only=True)
    if payload['base_sha256'] != base.WEIGHT_SHA or set(payload['train_values']) & set(payload['held_values']):
        raise ValueError('Writer or held token provenance mismatch')
    values = payload['held_values'][:8]
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    writers = {}
    for name in ('shallow_layer_1', 'final_normalized'):
        writer = TextSemanticWriter(model.config.hidden_size).to('cuda')
        writer.load_state_dict(payload['weights'][f'93471:{name}'], strict=True)
        writers[name] = writer.eval()
    install_sequence_scaled(model, scale=32.)
    rows = []
    for person in prior.HELD_NAMES:
        question = f'{person}喜欢喝什么饮料？'
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        for value in values:
            fact = f'{person}喜欢喝{value}。'
            encoded, positions = value_positions(tokenizer, fact, value)
            if len(positions) != 1:
                raise ValueError('Expected one value token')
            encoded = {k: v.to('cuda') for k, v in encoded.items()}
            token = int(encoded['input_ids'][0, positions[0]])
            with torch.inference_mode():
                outputs = model.model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
                vectors = {'shallow_layer_1': writers['shallow_layer_1'](outputs.hidden_states[1][0, positions[0]].float()),
                           'final_normalized': writers['final_normalized'](outputs.last_hidden_state[0, positions[0]].float()),
                           'direct_embedding': model.model.embed_tokens.weight[token].float()}
                eos = model.model.embed_tokens.weight[tokenizer.eos_token_id].float()
                generated = {}
                for mode, vector in vectors.items():
                    sequence = torch.stack((vector, eos))
                    with use_slot_sequence(sequence):
                        result = model.generate(**prompt, use_cache=True, do_sample=False, max_new_tokens=4,
                                                pad_token_id=tokenizer.pad_token_id,
                                                eos_token_id=tokenizer.eos_token_id)
                    tail = result[0, prompt['input_ids'].shape[1]:].tolist()
                    generated[mode] = {'ids': tail, 'text': tokenizer.decode(tail, skip_special_tokens=True),
                                       'exact': tail == [token, tokenizer.eos_token_id]}
            rows.append({'person': person, 'value': value, 'token': token, 'generated': generated})
        print(json.dumps({'completed_person': person, 'cases': len(rows)}, ensure_ascii=False), flush=True)
    if base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA or base.sha(source_path) != source_sha:
        raise ValueError('Source changed')
    summary = {mode: sum(row['generated'][mode]['exact'] for row in rows)
               for mode in ('shallow_layer_1', 'final_normalized', 'direct_embedding')}
    report = {'base_sha256': base.WEIGHT_SHA, 'writer_sha256': source_sha, 'rows': rows, 'summary': summary,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two held names and eight single-character held tokens, synthetic fixed sentence.',
                              'Boundaries and identity supplied by the experiment, not autonomous routing.',
                              'The sequence bridge directly injects values and EOS, not learned language retrieval.',
                              'Unmodified base; no replacement layer or persistent individual.',
                              'Restricted exact token metric; no multi-token, open dialogue or unseen-domain evidence.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'summary': summary, 'cases': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
