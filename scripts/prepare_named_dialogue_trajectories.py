"""Collect student reply prefixes and score them with a local instruction model."""

import argparse
import random
from pathlib import Path

from scripts.deepseek_spora_full_stream_loader import load_qwen_streamed
from scripts.train_named_memory_dialogue import MODEL, SEED, examples, load_full_streamed, sha
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import atomic_save
import torch
from transformers import AutoTokenizer


ORDINARY = 151643
DONOR_SHA = 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'


@torch.inference_mode()
def collect(checkpoint, tokenizer, groups, count):
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    decoder = runtime.decoder
    rows = groups['varied']
    selected = random.Random(SEED + 7).sample(range(len(rows)), count)
    result = []
    for index in selected:
        prompt, target, *_ = rows[index]
        inputs = torch.tensor([prompt], device='cuda')
        result_tokens = decoder.generate(inputs, attention_mask=torch.ones_like(inputs), use_cache=True,
                                         do_sample=False, max_new_tokens=min(len(target) + 2, 12),
                                         temperature=None, top_p=None, top_k=None,
                                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        generated = []
        for token in result_tokens[0, len(prompt):].tolist():
            generated.append(token)
            if token >= ORDINARY:
                break
        result.append({'index': index, 'prompt': prompt, 'target': target, 'generated': generated})
        if len(result) % 32 == 0:
            print('Student trajectories', len(result), '/', count, flush=True)
    return result


@torch.inference_mode()
def score(donor, tokenizer, instructor, groups, collected):
    ordinary_vocab = [{text: index for text, index in encoder.get_vocab().items() if index < ORDINARY}
                      for encoder in (tokenizer, instructor)]
    if ordinary_vocab[0] != ordinary_vocab[1]:
        raise ValueError('Student and teacher must share ordinary token coordinates')
    model = load_qwen_streamed(donor, DONOR_SHA)
    result = []
    for row in collected:
        index, generated = row['index'], row['generated']
        prompt, target, _, _, request = groups['varied'][index]
        teacher_prompt, _ = request
        if row['prompt'] != prompt or row['target'] != target or not generated:
            raise ValueError('Student trajectory does not match the training request')
        if any(token >= ORDINARY for token in generated[:-1]):
            raise ValueError('A trajectory contains an unsupported internal special token')
        positions = sorted({0, len(generated) // 3, 2 * len(generated) // 3, len(generated) - 1})
        offsets = torch.tensor(positions, device='cuda')
        ids = torch.tensor([teacher_prompt + generated[:-1]], device='cuda')
        logits = model(input_ids=ids, use_cache=False,
                       logits_to_keep=offsets + len(teacher_prompt) - 1).logits[0].float()
        eos = max(model.generation_config.eos_token_id)
        compact = torch.cat((logits[:, :ORDINARY], logits[:, eos:eos + 1]), dim=-1)
        scores, choices = compact.topk(32, dim=-1)
        choices[choices == ORDINARY] = tokenizer.eos_token_id
        result.append({**row, 'positions': positions, 'choices': choices.cpu(),
                       'log_probs': (scores / 2).log_softmax(-1).cpu()})
        if len(result) % 32 == 0:
            print('Scored trajectories', len(result), '/', len(collected), flush=True)
    return result


def main(checkpoint, output, donor, trajectories, count):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    digest = sha(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if donor is None:
        groups = examples(tokenizer, varied=True)
        data = {'checkpoint_sha256': digest, 'rows': collect(checkpoint, tokenizer, groups, count)}
    else:
        donor_files = tuple(donor / name for name in ('model.safetensors', 'config.json', 'tokenizer.json',
                                                      'tokenizer_config.json', 'generation_config.json'))
        sources = {str(path): sha(path) for path in (trajectories, *donor_files)}
        recorded = torch.load(trajectories, map_location='cpu', weights_only=True)
        if recorded['checkpoint_sha256'] != digest:
            raise ValueError('Collected trajectories belong to a different model')
        instructor = AutoTokenizer.from_pretrained(donor, local_files_only=True, trust_remote_code=False)
        groups = examples(tokenizer, varied=True, instructor=instructor)
        data = {'checkpoint_sha256': digest, 'sources': sources,
                'rows': score(donor, tokenizer, instructor, groups, recorded['rows'])}
        if any(sha(path) != sources[str(path)] for path in (trajectories, *donor_files)):
            raise ValueError('A teacher phase input changed')
    if sha(checkpoint) != digest:
        raise ValueError('Student checkpoint changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(data, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('checkpoint', 'output'):
        parser.add_argument(f'--{argument}', type=Path, required=True)
    parser.add_argument('--donor', type=Path)
    parser.add_argument('--trajectories', type=Path)
    parser.add_argument('--count', type=int, default=128)
    args = parser.parse_args()
    if (args.donor is None) != (args.trajectories is None) or not 1 <= args.count <= 409:
        parser.error('Provide a donor together with collected trajectories, or choose a collection size from 1 to 409')
    main(**vars(args))
