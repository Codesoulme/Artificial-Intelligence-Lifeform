"""Probe named and temporal natural queries on an isolated compatible individual."""

import argparse
import copy
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import KEYS, NAMES, pooled, predict
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_original_layer_name_router import LOCATOR_PROMPT, SequenceLocator, SpanLocator, choose_span, lookahead
from scripts.deepseek_spora_runtime_chat import _prepare_memory_read
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import grow
from spora.identity_fingerprints import IdentityFingerprints
from spora.identity_read_guard import IdentityReadGuard
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced
from spora.language_training import atomic_save, frozen_digest
from spora.memory_output import MemoryOutputAdapter
from spora.named_address import address_basis_sha256, feature_files_sha256


def controlled(organism, offset):
    state = fresh_controlled_state(organism)
    for index, key in enumerate(KEYS):
        for revision in (0, 1):
            if state.cursor >= len(state.pages) * organism.config.time_count:
                state = grow(organism, state)
            value = (index + offset + 3 * revision) % len(WORDS)
            state, decision = observe_sourced(organism, state, key, value, 'direct', 1., valid_at=state.cursor)
            if not decision['current_write']:
                raise ValueError('Controlled observation did not update memory')
    return state


def load_head(folder, filename, digest_key, checkpoint_sha):
    path = folder / filename
    info = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    digest = sha(path)
    if info[digest_key] != digest:
        raise ValueError(f'{filename} does not match its training report')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload['checkpoint_sha256'] != checkpoint_sha or payload['base_sha256'] != WEIGHT_SHA
            or payload['feature_config_sha256'] != feature_files_sha256(MODEL)):
        raise ValueError(f'{filename} belongs to a different model')
    return payload, digest


def cases():
    questions = [(f'现在的记录中，{name}最爱哪种饮品？', index, 0)
                 for index, name in enumerate(NAMES)]
    questions += [(f'在上一条记录更新以前，{name}通常喜欢喝什么？', index, 1)
                  for index, name in enumerate(NAMES)]
    questions += [(f'能不能查出{name}{suffix}之前常喝哪种饮料？', None, None)
                  for name, suffix in zip(NAMES, ('溪', '遥', '彦', '岑', '羽', '朗', '言', '川'), strict=True)]
    questions += [('小林和阿宁的饮品偏好各是什么？', None, None),
                  ('请给小梅写一段问候，别读取饮品记录。', None, None),
                  ('帮我计算三乘以四。', None, None)]
    return questions


def calibration_cases():
    return ([(f'你之前把{name}哪种饮品记下来了？', i)
             for i, name in enumerate(NAMES)]
            + [(f'想问一下，{name}最近爱喝什么饮品？', i)
               for i, name in enumerate(NAMES)]
            + [(f'我想知道{name}{suffix}选过的饮品。', None)
               for name, suffix in zip(NAMES, ('皓', '柏', '荷', '舟', '言', '澄', '亭', '珂'), strict=True)]
            + [('阿哲和小梅之前分别爱喝什么？', None)])


def answer(decoder, tokenizer, organism, question, state, key, mode):
    event = None
    if mode != 0:
        selection = select_event(organism, state, key, 'effective_previous' if mode == 1 else 'effective_latest',
                                 as_of=state.cursor - 1)
        if selection['status'] != 'experimental_candidate':
            return {'reason': selection['reason'], 'answer': None}
        event = selection['event_time']
    memory, evidence = _prepare_memory_read(organism, state, key, ('current', 'past', 'later')[mode], event)
    index = int(evidence.argmax(-1)[0])
    if index >= len(WORDS):
        return {'reason': 'unknown_value', 'answer': None}
    inputs = closed_think_inputs(tokenizer, question, 'cuda')
    generated = decoder.generate(**inputs, **memory, memory_output_logits=evidence,
                                 max_new_tokens=32, do_sample=False, top_p=None, temperature=None,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return {'reason': 'generated', 'answer': text, 'matrix_value': WORDS[index], 'event_time': event}


def fit_output(decoder, tokenizer, organism):
    """Calibrate the existing first-token adapter on frozen predictions."""
    training = (
        ('段晴', '备忘中{person}如今爱喝的东西是什么？', '早先登记了{person}爱喝哪一种？'),
        ('李樊', '请告诉我{person}当前饮品的名称。', '更正前{person}登记过哪种饮品？'),
    )
    spare_keys = [key for key in range(organism.config.key_count) if key not in KEYS]
    logits, memories, targets = [], [], []
    for key, (person, current, previous) in zip(spare_keys[:len(training)], training, strict=True):
        for value, word in enumerate(WORDS):
            state = fresh_controlled_state(organism)
            state, _ = observe_sourced(organism, state, key, value, 'direct', 1., valid_at=0)
            state, _ = observe_sourced(organism, state, key, (value + 1) % len(WORDS), 'direct', 1., valid_at=1)
            for question, mode, event, expected in ((current, 'current', None, WORDS[(value + 1) % len(WORDS)]),
                                                    (previous, 'past', 0, word)):
                with torch.no_grad():
                    controls, evidence = _prepare_memory_read(organism, state, key, mode, event)
                    if int(evidence.argmax(-1)[0]) != WORDS.index(expected):
                        raise ValueError('Training memory failed to recover the known value')
                    inputs = closed_think_inputs(tokenizer, question.format(person=person), 'cuda')
                    prediction = decoder(**inputs, **controls, logits_to_keep=1).logits[:, -1].float().cpu()
                logits.append(prediction)
                memories.append(evidence.float().detach().cpu())
                targets.append(tokenizer.encode(expected, add_special_tokens=False)[0])
    base_logits, memory_logits = torch.cat(logits), torch.cat(memories)
    target = torch.tensor(targets)
    original = copy.deepcopy(decoder.output_adapter).cpu().eval()
    adapter = copy.deepcopy(original).requires_grad_(True).train()
    with torch.no_grad():
        old_bias = original(memory_logits)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=.003, weight_decay=0.)
    losses = []
    for step in range(120):
        bias = adapter(memory_logits)
        loss = F.cross_entropy(base_logits + bias, target) + .001 * (bias - old_bias).square().sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step in (0, 29, 59, 119):
            losses.append(float(loss.detach()))
    adapter.eval().requires_grad_(False)
    with torch.no_grad():
        new_bias = adapter(memory_logits)
    stats = {'sequences': len(targets), 'losses': losses,
             'first_token_before': int((base_logits + old_bias).argmax(-1).eq(target).sum()),
             'first_token_after': int((base_logits + new_bias).argmax(-1).eq(target).sum())}
    return adapter, stats


@torch.inference_mode()
def score(decoder, tokenizer, organism, router, time_head, states, *, calibrate_address=False,
          oracle_diagnostic=False, trim_punctuation=False):
    questions = cases()
    calibration = calibration_cases() if calibrate_address else []
    texts = [question[0] for question in questions] + [question for question, _ in calibration]
    if len(set(texts)) != len(texts):
        raise ValueError('Calibration and evaluation questions must be distinct')
    hidden_indices = tuple(router['hidden_indices'])
    if hidden_indices != tuple(time_head['hidden_indices']):
        raise ValueError('Router heads require different features')
    user_prefix = router.get('user_prefix', '')
    if user_prefix not in ('', LOCATOR_PROMPT):
        raise ValueError('Unsupported locate prompt')
    features, masks = extract(decoder.base, tokenizer, texts, batch_size=1, hidden_indices=hidden_indices,
                              user_prefix=user_prefix)
    bases = torch.stack([F.normalize(site.current_basis.detach().cpu().float(), dim=-1)
                         for site in organism.memories])
    vectors = torch.cat([F.layer_norm(features[:, site].float(), (features.shape[-1],)) for site in range(2)], dim=-1)
    following = router.get('locator_following_tokens', 0)
    if following == 2:
        vectors = torch.stack([lookahead(row) for row in vectors.masked_fill(~masks[..., None], 0.)])
    elif following != 0:
        raise ValueError('Unsupported locator context')
    architecture = router.get('locator_architecture', 'linear')
    if architecture in ('span', 'sequence'):
        vectors = vectors.masked_fill(~masks[..., None], 0.)
        locator = (SpanLocator if architecture == 'span' else SequenceLocator)(vectors.shape[-1]).eval()
    elif architecture == 'linear':
        locator = nn.Linear(vectors.shape[-1], 1).eval()
    else:
        raise ValueError('Unsupported locator architecture')
    locator.load_state_dict(router['locator_state'], strict=True)
    locator_probs = (locator(vectors, masks) if architecture == 'span'
                     else locator(vectors).squeeze(-1).sigmoid())
    time_features, time_masks = (extract(decoder.base, tokenizer, texts[:len(questions)], batch_size=1,
                                         hidden_indices=hidden_indices) if user_prefix else (features, masks))
    pooled_questions = pool(time_features, time_masks)
    temporal = nn.Linear(pooled_questions.shape[-1], len(time_head['classes'])).eval()
    temporal.load_state_dict(time_head['state'], strict=True)
    temporal_probs = temporal(pooled_questions).softmax(-1)
    fingerprints = IdentityFingerprints(organism.config.key_count)
    for name, key in zip(NAMES, KEYS, strict=True):
        fingerprints.bind(key, name)
    guard = IdentityReadGuard(fingerprints, tokenizer)

    def propose(i, question, locator_threshold):
        selected, single = choose_span(locator_probs[i], masks[i], locator_threshold)
        if not single:
            return None, None, 'no_single_span', None
        guarded_question = user_prefix + question
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': guarded_question}], tokenize=False,
                                             add_generation_prompt=True)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        size = len(encoded['input_ids'])
        if bool(selected[size:].any()):
            return None, None, 'outside_question', None
        if trim_punctuation:
            selected = selected.clone()
            indices = selected.nonzero(as_tuple=True)[0].tolist()
            for ordered in (indices, reversed(indices)):
                for index in ordered:
                    left, right = encoded['offset_mapping'][index]
                    piece = text[left:right]
                    if not piece or not all(unicodedata.category(char).startswith('P') for char in piece):
                        break
                    selected[index] = False
            if not bool(selected.any()):
                return None, None, 'no_name_after_punctuation', None
        selected_text = guard.selected_name(guarded_question, selected[:size])
        proposed, similarity = predict(pooled(features[i:i+1], selected.unsqueeze(0)), router['projection'], bases)
        key, second = proposed[0].tolist()
        if key != second:
            return None, None, 'address_disagreement', selected_text
        if not guard.authorize(guarded_question, selected[:size], key):
            return None, None, 'identity_mismatch', selected_text
        return key, float(similarity.min()), None, selected_text

    threshold, locator_threshold = router['address_threshold'], router['locator_threshold']
    calibration_rows = []
    if calibration:
        best = (-1, -1, -1.)
        for candidate in sorted({locator_threshold, .25, .35, .45, .55, .65, .75, .85, .95}):
            candidate_rows = []
            for i, (question, expected_index) in enumerate(calibration, start=len(questions)):
                key, similarity, reason, selected_text = propose(i, question, candidate)
                expected_key = None if expected_index is None else KEYS[expected_index]
                candidate_rows.append({'question': question, 'expected_key': expected_key,
                                       'proposed_key': key, 'similarity': similarity, 'reason': reason,
                                       'selected_text': selected_text})
            false_scores = [row['similarity'] for row in candidate_rows
                            if row['proposed_key'] is not None and row['proposed_key'] != row['expected_key']]
            valid_scores = [row['similarity'] for row in candidate_rows
                            if row['proposed_key'] is not None and row['proposed_key'] == row['expected_key']]
            candidate_threshold = max(0., max(false_scores, default=-1.) + 1e-6,
                                      min(valid_scores, default=threshold))
            rank = (sum(score >= candidate_threshold for score in valid_scores), -len(false_scores),
                    -abs(candidate - router['locator_threshold']))
            if rank > best:
                best = rank
                threshold, locator_threshold, calibration_rows = candidate_threshold, candidate, candidate_rows

    outcomes = []
    for i, (question, expected_index, expected_mode) in enumerate(questions):
        expected_key = None if expected_index is None else KEYS[expected_index]
        mode = int(temporal_probs[i].argmax())
        if mode != 3 and float(temporal_probs[i, mode]) < time_head['threshold']:
            mode = 3
        outcome = {'question': question, 'expected_key': expected_key, 'expected_mode': expected_mode,
                   'mode': mode, 'key': None, 'answers': [], 'correct_pair': False}
        outcomes.append(outcome)
        expected = None
        if expected_index is not None:
            expected = [WORDS[(expected_index + offset + (0 if expected_mode == 1 else 3)) % len(WORDS)]
                        for offset in (0, 2)]
            outcome['expected_values'] = expected

        def correct_pair(answers):
            return all(item['answer'] is not None and expected[j] in item['answer']
                       and expected[1 - j] not in item['answer'] for j, item in enumerate(answers))

        key, similarity, reason, selected_text = propose(i, question, locator_threshold)
        if reason is None and mode == 3:
            reason = 'temporal_abstain'
        if reason is None and similarity < threshold:
            reason = 'address_similarity'
        outcome.update(proposed_key=key, similarity=similarity, routing_reason=reason, selected_text=selected_text)
        if reason is None:
            outcome['key'] = key
            outcome['answers'] = [answer(decoder, tokenizer, organism, question, state, key, mode) for state in states]
            if key == expected_key and mode == expected_mode:
                outcome['correct_pair'] = correct_pair(outcome['answers'])
        if oracle_diagnostic and expected_key is not None:
            oracle = outcome['answers'] if outcome['key'] == expected_key and mode == expected_mode else [
                answer(decoder, tokenizer, organism, question, state, expected_key, expected_mode) for state in states
            ]
            outcome['oracle_answers'] = oracle
            outcome['oracle_correct_pair'] = correct_pair(oracle)
    return outcomes, {'threshold': threshold, 'previous_threshold': router['address_threshold'],
                      'locator_threshold': locator_threshold, 'calibration_rows': calibration_rows}


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint_sha = sha(args.checkpoint)
    router, router_sha = load_head(args.router, 'router.pt', 'router_sha256', checkpoint_sha)
    temporal, temporal_sha = load_head(args.temporal, 'temporal.pt', 'temporal_sha256', checkpoint_sha)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder, organism, original = load_integrated(args.checkpoint, load_full_streamed(), MODEL)
    if address_basis_sha256(organism) != router['address_basis_sha256']:
        raise ValueError('Name router belongs to different address weights')
    decoder.eval().requires_grad_(False)
    original_digest = frozen_digest(organism, original)
    states = [controlled(organism, offset) for offset in (0, 2)]
    state_digests = [frozen_digest(organism, state) for state in states]
    output_fit = None
    if args.calibrate_output:
        torch.set_num_threads(2)
        adapter, output_fit = fit_output(decoder, tokenizer, organism)
        print(output_fit, flush=True)
        adapter_path = args.output.with_name('output-adapter.pt')
        if adapter_path.exists():
            raise FileExistsError(adapter_path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_save({'checkpoint_sha256': checkpoint_sha, 'state': adapter.state_dict(),
                     'config': adapter.config_dict()}, adapter_path)
        output_fit['adapter_sha256'] = sha(adapter_path)
        decoder.output_adapter = adapter.to(next(decoder.parameters()).device)
    elif args.output_adapter is not None:
        path = args.output_adapter / 'output-adapter.pt'
        prior = json.loads((args.output_adapter / 'result.json').read_text(encoding='utf-8'))
        output_fit = prior['output_fit']
        if sha(path) != output_fit['adapter_sha256']:
            raise ValueError('Output adapter weight digest mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError('Output adapter belongs to a different model')
        adapter = MemoryOutputAdapter(**payload['config'])
        adapter.load_state_dict(payload['state'], strict=True)
        decoder.output_adapter = adapter.eval().requires_grad_(False).to(next(decoder.parameters()).device)
    rows, address_fit = score(decoder, tokenizer, organism, router, temporal, states,
                              calibrate_address=args.calibrate_address, oracle_diagnostic=args.oracle_diagnostic,
                              trim_punctuation=args.trim_punctuation)
    if (original_digest != frozen_digest(organism, original) or sha(args.checkpoint) != checkpoint_sha
            or state_digests != [frozen_digest(organism, state) for state in states]):
        raise ValueError('Evaluation changed memory')
    known = [row for row in rows if row['expected_key'] is not None]
    unsupported = [row for row in rows if row['expected_key'] is None]
    summary = {'known': len(known), 'correct_key': sum(r['expected_key'] == r['key'] for r in known),
               'correct_time': sum(r['expected_mode'] == r['mode'] for r in known),
               'correct_pairs': sum(r['correct_pair'] for r in known), 'unsupported': len(unsupported),
               'unsupported_routed': sum(r['key'] is not None for r in unsupported)}
    if args.oracle_diagnostic:
        summary['oracle_correct_pairs'] = sum(r['oracle_correct_pair'] for r in known)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'summary': summary, 'rows': rows, 'checkpoint_sha256': checkpoint_sha,
                   'router_sha256': router_sha, 'temporal_sha256': temporal_sha,
                   'output_fit': output_fit, 'address_fit': address_fit}, stream,
                  ensure_ascii=False, indent=2)
        stream.write('\n')
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'router', 'temporal', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    output = parser.add_mutually_exclusive_group()
    output.add_argument('--calibrate-output', action='store_true')
    output.add_argument('--output-adapter', type=Path)
    parser.add_argument('--calibrate-address', action='store_true')
    parser.add_argument('--oracle-diagnostic', action='store_true')
    parser.add_argument('--trim-punctuation', action='store_true')
    main(parser.parse_args())
