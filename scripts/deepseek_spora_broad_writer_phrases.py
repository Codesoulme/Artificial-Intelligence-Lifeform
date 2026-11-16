"""Check a broad single-token writer on multi-token, previously used drink names."""
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
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from scripts.deepseek_spora_output_aware_writer import GAIN, norm_vectors
from spora.research_semantic_slot import install_sequence_scaled, use_slot_sequence
from spora.semantic_text_writer import TextSemanticWriter, value_positions

SOURCE = ROOT / 'runs/deepseek-spora-output-aware-broad-20261116'
OUT = ROOT / 'runs/deepseek-spora-broad-writer-phrases-20261116.json'
VALUES = ('酸奶', '咖啡', '豆浆', '矿泉水', '柠檬水', '椰汁', '红茶', '苹果汁')
QUESTIONS = ('{name}喜欢喝什么饮料？', '请只说{name}最喜欢的饮品。')


def main(output=OUT, replace_spora=False):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    report = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    source = SOURCE / 'writer.pt'
    digest = base.sha(source)
    if report['base_sha256'] != base.WEIGHT_SHA or report['writer_sha256'] != digest:
        raise ValueError('Writer provenance mismatch')
    payload = torch.load(source, map_location='cpu', weights_only=True)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    paths = {i: PAIR / f'layer-{i}.pt' for i in (6, 12)} if replace_spora else {}
    replacement_hashes = {str(i): base.sha(path) for i, path in paths.items()}
    model = load_full_streamed(skip_replaced=(6, 12) if replace_spora else ()).eval()
    for index, path in paths.items():
        model.model.layers[index] = _load_candidate(path, index, model).eval()
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError('Unmaterialized model parameter')
    writer = TextSemanticWriter(model.config.hidden_size).to('cuda').eval()
    writer.load_state_dict(payload['state'], strict=True)
    install_sequence_scaled(model, scale=GAIN)
    train_tokens = {tokenizer.encode(value, add_special_tokens=False)[0] for value in report['train_values']}
    rows = []
    for name in prior.HELD_NAMES:
        for value in VALUES:
            encoded, positions = value_positions(tokenizer, f'{name}喜欢喝{value}。', value)
            encoded = {key: tensor.to('cuda') for key, tensor in encoded.items()}
            token_ids = encoded['input_ids'][0, positions].tolist()
            with torch.inference_mode():
                hidden = model.model(**encoded, use_cache=False, output_hidden_states=True,
                                     return_dict=True).hidden_states[1][0, positions].float()
                learned = norm_vectors(writer(hidden), payload['train_norm'])
                direct = model.model.embed_tokens.weight[token_ids].float()
                eos = model.model.embed_tokens.weight[tokenizer.eos_token_id].float().unsqueeze(0)
                for template in QUESTIONS:
                    question = template.format(name=name)
                    prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
                    variants = {}
                    for mode, slots in (('learned', learned), ('direct', direct)):
                        with use_slot_sequence(torch.cat((slots, eos), dim=0)):
                            generated = model.generate(**prompt, use_cache=True, do_sample=False,
                                                       max_new_tokens=len(token_ids) + 3,
                                                       pad_token_id=tokenizer.pad_token_id,
                                                       eos_token_id=tokenizer.eos_token_id)
                        tail = generated[0, prompt['input_ids'].shape[1]:].tolist()
                        variants[mode] = {'ids': tail, 'text': tokenizer.decode(tail, skip_special_tokens=True),
                                          'exact': tail == token_ids + [tokenizer.eos_token_id]}
                    rows.append({'name': name, 'value': value, 'question': question, 'value_ids': token_ids,
                                 'train_target_overlap': [token for token in token_ids if token in train_tokens],
                                 'variants': variants})
        print(json.dumps({'completed_name': name, 'cases': len(rows)}, ensure_ascii=False), flush=True)
    if (base.sha(source) != digest or base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or any(base.sha(path) != replacement_hashes[str(index)] for index, path in paths.items())):
        raise ValueError('Protected input changed')
    summary = {mode: sum(row['variants'][mode]['exact'] for row in rows) for mode in ('learned', 'direct')}
    result = {'rows': rows, 'summary': summary, 'cases': len(rows), 'base_sha256': base.WEIGHT_SHA,
              'writer_sha256': digest, 'replacements': replacement_hashes, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two held names, eight previously studied drink phrases and two questions.',
                              'Value token positions and answer EOS are externally supplied.',
                              'Some phrase tokens may have been supervised individually or used as negatives.',
                              'Direct-vector bridge is an oracle control, not language understanding.',
                              'Oracle bridge; no persistent individual or natural memory routing.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'summary': summary, 'cases': len(rows)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--replace-spora', action='store_true')
    args = parser.parse_args()
    main(args.output, args.replace_spora)
