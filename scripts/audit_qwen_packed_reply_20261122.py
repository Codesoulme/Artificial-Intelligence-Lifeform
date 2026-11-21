"""Save the learned whole-value reply adapter and verify native reload behavior."""

import argparse
import json
from pathlib import Path

import torch

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, evaluate, load_reply_head, records
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import frozen_digest
from spora.memory_reply import MemoryReplyAdapter
from transformers import AutoTokenizer

ORDINARY = ('用一句话解释彩虹是怎么出现的。', '说出三个适合散步的地方。', '七加八等于多少？')


@torch.inference_mode()
def ordinary_outputs(decoder, tokenizer):
    rows = []
    for question in ORDINARY:
        inputs = closed_think_inputs(tokenizer, question, 'cuda')
        generated = decoder.generate(**inputs, max_new_tokens=24, do_sample=False, use_cache=True,
                                     temperature=None, top_p=None, pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id)
        rows.append(generated[0, inputs['input_ids'].shape[1]:].tolist())
    return rows


def compare_payloads(original, packed):
    before, after = (torch.load(path, map_location='cpu', weights_only=True) for path in (original, packed))
    left, right = ({k: v for k, v in item['metadata'].items() if k != 'output_adapter'} for item in (before, after))
    if left != right:
        raise ValueError('Packing changed metadata outside the output adapter')
    left, right = ({k: v for k, v in item['tensors'].items() if not k.startswith('output/')}
                   for item in (before, after))
    if left.keys() != right.keys() or any(not torch.equal(value, right[key]) for key, value in left.items()):
        raise ValueError('Packing changed tensors outside the output adapter')


def main(checkpoint, reply, output, *, verify=False):
    output_path = output / 'packed.pt'
    manifest_path = output / 'packed.json'
    verification = output / 'verified.json'
    target = verification if verify else manifest_path
    if target.exists() or (not verify and output_path.exists()):
        raise FileExistsError(target)
    torch.set_num_threads(2)
    digest = sha(checkpoint)
    adapters, reply_sha = load_reply_head(reply, digest, 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    output.mkdir(parents=True, exist_ok=True)
    if verify:
        recorded = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (recorded['source_sha256'] != digest or recorded['reply_sha256'] != reply_sha
                or recorded['packed_sha256'] != sha(output_path)):
            raise ValueError('Source or candidate changed before native reload')
    runtime = load_integrated_runtime(output_path if verify else checkpoint, load_full_streamed(), MODEL)
    decoder = runtime.decoder.eval().requires_grad_(False)
    state_digest = frozen_digest(runtime.organism, runtime.state)
    evidence_outputs = ordinary_outputs(decoder, tokenizer)
    if verify:
        compare_payloads(checkpoint, output_path)
        expected = json.loads((reply / 'result.json').read_text(encoding='utf-8'))['results']['sequence']
        result = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
        report = {'packed_sha256': sha(output_path), 'ordinary_exact': evidence_outputs == recorded['ordinary_tokens'],
                  'reply_exact': result == expected, 'reply': result, 'individual_sha256': state_digest}
        if not report['ordinary_exact'] or not report['reply_exact'] or state_digest != recorded['individual_sha256']:
            raise ValueError('Native checkpoint generation differs after reload')
    else:
        config = adapters[0].config_dict()
        adapter = MemoryReplyAdapter(**config, steps=len(adapters))
        adapter.heads.load_state_dict(adapters.state_dict(), strict=True)
        decoder.output_adapter = adapter.to('cuda').eval().requires_grad_(False)
        if ordinary_outputs(decoder, tokenizer) != evidence_outputs:
            raise ValueError('Attaching the reply head changed ordinary generation')
        runtime.save(output_path)
        compare_payloads(checkpoint, output_path)
        report = {'source_sha256': digest, 'reply_sha256': reply_sha, 'packed_sha256': sha(output_path),
                  'ordinary_tokens': evidence_outputs, 'individual_sha256': state_digest}
    if sha(checkpoint) != digest or frozen_digest(runtime.organism, runtime.state) != state_digest:
        raise ValueError('Protected source changed during checkpoint audit')
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('Packed reply checkpoint', report['packed_sha256'], 'verified', verify, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'reply', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    main(args.checkpoint, args.reply, args.output, verify=args.verify)
