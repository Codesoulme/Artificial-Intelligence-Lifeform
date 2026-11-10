"""Compare complete yes/no continuations under opposite coreference questions."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.audit_mmc_pronoun_pairs import DEV, DEV_SHA, load
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_mmc_choice_baseline import balanced_subset
from scripts.deepseek_mmc_pair_probe import score, select_pairs
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load as load_individual
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-mmc-sequence-baseline-20261110.json'
ANSWERS = ('是。', '否。')


def make_prompt(text, ask_different):
    if type(ask_different) is not bool:
        raise ValueError('Question polarity must be boolean')
    question = '不同的人' if ask_different else '同一个人'
    return ('对话中的数字表示发言者。判断<m1>和<m2>标记的代词是否指' + question
            + '，只回答是或否。\n' + text + '\n答案：')


def sequence_logprob(logits, token_ids, prefix_length):
    if (logits.ndim != 2 or token_ids.ndim != 1 or logits.shape[0] != token_ids.shape[0]
            or type(prefix_length) is not int or not 1 <= prefix_length < len(token_ids)):
        raise ValueError('Invalid sequence scoring inputs')
    selected = logits[prefix_length - 1:-1].float().log_softmax(-1)
    target = token_ids[prefix_length:]
    result = selected.gather(1, target[:, None]).sum()
    if not bool(torch.isfinite(result)):
        raise ValueError('Nonfinite answer score')
    return float(result)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'individual': INDIVIDUAL, 'dev': DEV}
    before = {name: sha(path) for name, path in files.items()}
    if before != {'weight': WEIGHT_SHA, 'individual': INDIVIDUAL_SHA, 'dev': DEV_SHA}:
        raise ValueError('Protected source mismatch')
    rows = balanced_subset(select_pairs(load(DEV, DEV_SHA), 30, current_sentence_tail=True))
    config_sha = feature_files_sha256(MODEL)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load_individual(INDIVIDUAL, 'cpu')
    digest = frozen_digest(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    answers = [tokenizer.encode(answer, add_special_tokens=False) for answer in ANSWERS]
    if any(not ids for ids in answers) or answers[0] == answers[1]:
        raise ValueError('Expected distinct nonempty answers')
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    predictions = [[], []]
    results = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            scores_by_question = []
            for inverted in (False, True):
                inputs = closed_think_inputs(tokenizer, make_prompt(row['text'], inverted), 'cuda')
                prefix = inputs['input_ids'][0]
                scores = []
                for answer in answers:
                    ids = torch.cat((prefix, torch.tensor(answer, device=prefix.device)))
                    logits = base(input_ids=ids[None, :], use_cache=False).logits[0]
                    scores.append(sequence_logprob(logits, ids, len(prefix)))
                answer_index = int(scores[1] > scores[0])
                predictions[int(inverted)].append(int(answer_index == int(inverted)))
                scores_by_question.append(scores)
            results.append({'sample_sha256': row['sample_sha256'], 'label': row['label'],
                            'predicted': [predictions[0][-1], predictions[1][-1]], 'logprobs': scores_by_question})
            print(f'evaluated {index + 1}/{len(rows)}', flush=True)
    if (any(sha(path) != before[name] for name, path in files.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, original) != digest or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected source changed')
    report = {'status': 'sequence_scoring_development_only', 'samples': results,
              'scores': {'same_question': score(rows, predictions[0]), 'different_question': score(rows, predictions[1]),
                         'heuristic': score(rows, [r['heuristic'] for r in rows]),
                         'polarity_agreement': sum(a == b for a, b in zip(*predictions))},
              'answer_token_counts': [len(ids) for ids in answers], 'component_sha256': before,
              'feature_config_sha256': config_sha, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Answer scores sum log probabilities of forced tokens; no EOS or length normalization.',
                              'Two answer strings do not represent all possible answers; forced scoring is not free generation.',
                              'Reasoning section is forcibly closed; question polarity changes wording.',
                              'Twenty-four balanced pairs from previously inspected development scenes; not final evaluation.',
                              'Mention positions and speaker labels are supplied by the dataset.',
                              'Different annotated clusters may reflect incomplete or incorrect annotations.',
                              'Frozen base only; no SPORA memory, identity verification, or write decisions evaluated.',
                              'Subtitle data rights and pretraining overlap remain unknown.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report['scores']), flush=True)


if __name__ == '__main__':
    run()
