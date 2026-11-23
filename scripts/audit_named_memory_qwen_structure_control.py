"""Compare one intact instruction backbone before and after SPORA integration."""

import argparse
import json
from functools import partial
from pathlib import Path

from scripts.audit_local_named_chat import CASES, FRESH_CASES, run
from scripts.audit_spora_chat_backbones import reply
from scripts.deepseek_local_baseline import WEIGHT_SHA, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_qwen_streamed
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, TRAIN, WORDS, cache, evaluate, fit, make_adapters, records
from spora.integrated_checkpoint import OUTPUT_ADAPTER_FORMAT, _digest
from spora.integrated_open_base import IntegratedOpenBase, NativeSPORAMemoryReader
from spora.language_training import atomic_save, frozen_digest
from spora.memory_reply import MemoryReplyAdapter
from spora.pretrained_attention import SPORAResidualBlock
from spora.state_codec import restore
from transformers import AutoTokenizer


CHECKPOINT_SHA = 'f1d86b67b678e0fcbb2deab041817aca2f236db2ad898804236df508e2e7a4b8'
QWEN_SHA = 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'


def attach_control(base, metadata, tensors):
    reader_config = metadata['reader']
    if (metadata['format'] != OUTPUT_ADAPTER_FORMAT or metadata['base_sha256'] != WEIGHT_SHA
            or reader_config['kind'] != 'native-spora-delta-site'
            or any(name.startswith(('replacement/', 'value_position/', 'query_key_position/', 'position/'))
                   for name in tensors)):
        raise ValueError('Expected the DeepSeek named memory control configuration')
    individual = {name.removeprefix('individual/'): tensor for name, tensor in tensors.items()
                  if name.startswith('individual/')}
    organism, state = restore(metadata['individual_schema'], individual, 'cpu')
    reader = NativeSPORAMemoryReader(reader_config['hidden_width'], reader_config['memory_width'],
                                     reader_config['strength'])
    reader.load_state_dict({name.removeprefix('reader/'): tensor for name, tensor in tensors.items()
                            if name.startswith('reader/')}, strict=True)
    output_config = dict(metadata['output_adapter'])
    if output_config.pop('format') != MemoryReplyAdapter.format:
        raise ValueError('Expected the whole-value memory reply adapter')
    output = MemoryReplyAdapter(**output_config)
    output.load_state_dict({name.removeprefix('output/'): tensor for name, tensor in tensors.items()
                            if name.startswith('output/')}, strict=True)
    device = next(base.parameters()).device
    decoder = IntegratedOpenBase(base, reader.to(device), reader_config['layers'],
                                 output_adapter=output.to(device)).eval()
    decoder.migrate_pretrained_blocks(metadata['compatible_layers'])
    decoder.migrate_position_encoder(metadata['position_table_max_position'])
    for layer, fraction in metadata['feedforward_fractions'].items():
        index = int(layer)
        decoder.use_feedforward_memory(index, fraction=fraction,
                                       at_output=index in metadata['feedforward_output_layers'])
    for spec in metadata['residuals']:
        index, rank = spec['index'], spec['rank']
        if not spec.get('memory_only') or index in decoder.layer_indices:
            raise ValueError('Unexpected residual in the control checkpoint')
        layers = decoder.base.model.layers
        block = SPORAResidualBlock(layers[index], rank, memory_context=decoder._memory_context).eval()
        with torch.no_grad():
            block.down.weight.copy_(tensors[f'residual/{index}/down'])
            block.up.weight.copy_(tensors[f'residual/{index}/up'])
        layers[index] = block
    return decoder.requires_grad_(False), organism, state


def memory_inputs(tokenizer, question, device):
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                          add_generation_prompt=True)
    return tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)


def dialogue(model, tokenizer, eos):
    generate = partial(reply, model, tokenizer, device='cuda', deepseek=False, limit=96, eos=eos)

    def answer(history, question):
        return generate(history, question)['answer']

    return {label: run(model, tokenizer, cases=cases, reply_with_history=answer)
            for label, cases in (('original', CASES), ('fresh', FRESH_CASES))}


def load_calibration(path, reference, sources, decoder, organism):
    expected = reference['recalibration']
    digest = sha(path)
    if expected['adapter_sha256'] != digest or reference['sources'] != sources:
        raise ValueError('Calibration report does not match the supplied inputs')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload['donor_sha256'] != QWEN_SHA or payload['checkpoint_sha256'] != CHECKPOINT_SHA
            or payload['vocab_size'] != decoder.base.config.vocab_size
            or payload['value_count'] != organism.config.value_count):
        raise ValueError('Calibration weights do not match this control')
    module = MemoryReplyAdapter(payload['value_count'], payload['vocab_size'], payload['steps']).cuda().eval()
    module.heads.load_state_dict(payload['state'], strict=True)
    return module.requires_grad_(False), digest


def main(checkpoint, donor, output, updates, verify):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.manual_seed(20261124)
    files = (checkpoint, *(donor / name for name in ('model.safetensors', 'config.json', 'tokenizer.json',
                                                    'tokenizer_config.json', 'generation_config.json')))
    sources = {path: sha(path) for path in files}
    source_hashes = {str(path): digest for path, digest in sources.items()}
    if sources[checkpoint] != CHECKPOINT_SHA or sources[donor / 'model.safetensors'] != QWEN_SHA:
        raise ValueError('Expected pinned control inputs')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    metadata, tensors = payload['metadata'], payload['tensors']
    if _digest(metadata, tensors) != payload['integrity_sha256']:
        raise ValueError('Checkpoint integrity mismatch')
    tokenizer = AutoTokenizer.from_pretrained(donor, local_files_only=True, trust_remote_code=False)
    base = load_qwen_streamed(donor, QWEN_SHA)
    eos = base.generation_config.eos_token_id
    raw = None if verify is not None else dialogue(base, tokenizer, eos)
    if raw is not None:
        print('Native', [(value['cases_passed'], value['turns_passed']) for value in raw.values()], flush=True)
        torch.cuda.empty_cache()
    decoder, organism, state = attach_control(base, metadata, tensors)
    before = frozen_digest(organism, state)
    if verify is not None:
        reference = json.loads(verify.read_text(encoding='utf-8'))
        path = verify.with_suffix('.reply.pt')
        decoder.output_adapter, digest = load_calibration(path, reference, source_hashes, decoder, organism)
    after = dialogue(decoder, tokenizer, eos)
    print('Migrated', [(value['cases_passed'], value['turns_passed']) for value in after.values()], flush=True)
    torch.cuda.empty_cache()
    held = records(organism, HELD)
    memory = evaluate(decoder, tokenizer, organism, held, encode=memory_inputs, eos=eos)
    result = {'sources': source_hashes, 'individual_sha256': before, 'migrated': after, 'memory': memory}
    if verify is None:
        result.update(native=raw, dialogue_exact=raw == after)
    else:
        if (reference['migrated'] != after or reference['recalibration']['memory'] != memory
                or reference['individual_sha256'] != before or sha(path) != digest):
            raise ValueError('Reloaded control differs from the original experiment')
        result.update(reloaded=True, adapter_sha256=digest)
    if updates:
        output.parent.mkdir(parents=True, exist_ok=True)
        adapter_path = output.with_suffix('.reply.pt')
        if adapter_path.exists():
            raise FileExistsError(adapter_path)
        train = records(organism, TRAIN)
        if {row[0] for row in train} & {row[0] for row in held}:
            raise ValueError('Memory training and evaluation questions overlap')
        targets = [tokenizer.encode(word, add_special_tokens=False) + [tokenizer.eos_token_id] for word in WORDS]
        data = cache(decoder, tokenizer, organism, train, targets, encode=memory_inputs)
        steps = max(map(len, targets))
        vocab, values = base.config.vocab_size, organism.config.value_count
        adapters = make_adapters(vocab, values, steps).cuda()
        trace = fit(adapters, *data, updates)
        del data
        adapters.requires_grad_(False).eval()
        atomic_save({'donor_sha256': QWEN_SHA, 'checkpoint_sha256': CHECKPOINT_SHA,
                     'vocab_size': vocab, 'value_count': values, 'steps': steps,
                     'state': {name: tensor.cpu() for name, tensor in adapters.state_dict().items()}}, adapter_path)
        output_adapter = MemoryReplyAdapter(values, vocab, steps).cuda().eval()
        output_adapter.heads.load_state_dict(adapters.state_dict(), strict=True)
        decoder.output_adapter = output_adapter.requires_grad_(False)
        result['recalibration'] = {'updates': updates, 'loss': trace, 'adapter_sha256': sha(adapter_path),
                                   'memory': evaluate(decoder, tokenizer, organism, held, encode=memory_inputs, eos=eos)}
        print('Recalibrated memory', result['recalibration']['memory']['summary'], flush=True)
    if frozen_digest(organism, state) != before or any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('Control changed a protected input')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('Memory', memory['summary'], 'reload verified' if verify else 'native dialogue exact',
          result.get('dialogue_exact', True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('checkpoint', 'donor', 'output'):
        parser.add_argument(f'--{argument}', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--updates', type=int, default=0)
    modes.add_argument('--verify', type=Path, help='Reproduce a report using its saved calibration weights')
    args = parser.parse_args()
    if args.updates < 0:
        parser.error('Updates must be nonnegative')
    main(**vars(args))
