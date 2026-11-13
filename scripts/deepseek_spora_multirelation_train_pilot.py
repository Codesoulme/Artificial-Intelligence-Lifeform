"""Train existing memory readers on multiple controlled people and relations."""
from dataclasses import replace
import json
import math
import os
import random
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_bridge_reload_audit as audit
from spora.continuous_text_memory import ContinuousTextCodec
from spora.research_oracle_bridge import context_oracle_embeddings
from spora.research_pretrained_memory_attention import PretrainedMemoryAttention, disable_memory_attention

torch, p, s = audit.torch, audit.p, audit.s
audit.SOURCE = ROOT / 'runs/continuous-reader-cross-attention-pilot-20261113'
audit.OUT = ROOT / 'runs/deepseek-spora-multirelation-train-pilot-20261113'
audit.EXTRA_ONLY = True
audit.PARITY_VALUE_INDICES = (0, 1, 3, 9, 16, 17)
NAMES = ('小林', '小周', '小王')
HELD_NAME = '小赵'
DOMAINS = {
    'drink': ('喜欢喝', ('请只回答{name}喜欢喝什么饮品。', '{name}喜欢喝什么？只说饮品名称。'),
              ('茶', '咖啡'), ('牛奶', '豆浆')),
    'color': ('喜欢的颜色是', ('请只回答{name}喜欢的颜色。', '{name}喜欢哪种颜色？只写颜色。'),
              ('红色', '蓝色'), ('绿色', '黄色')),
    'hobby': ('喜欢的活动是', ('请只回答{name}喜欢的活动。', '{name}喜欢做什么活动？只写活动。'),
              ('读书', '跑步'), ('画画', '唱歌')),
}
STEPS = 144
HELD_VALUE_Q = 1
EVAL_ALTERNATIVE_QUESTION = False
EVAL_NOVEL_QUESTION = False
EVAL_SEEN_COUNT = None
CROSS_EVALUATION = False
INIT_FROM = None
EXPECTED_INITIAL_RESULT = None
REPLAY_FACTOR = 1
REPLAY_OLD_COUNT = 6
TEACHER_KL_WEIGHT = 0.0
TEACHER_KL_TEMPERATURE = 2.0
COUNTERFACTUAL_WEIGHT = 0.0
BRIDGE_MODE = 'learned'
MEMORY_ATTENTION = False
MEMORY_PROJECTION_LR = None
MEMORY_GAIN_LR = None
EVAL_ATTENTION_ABLATION = False
RELOAD_FROM = None
done = False


def checkpoint_due(completed, total, interval=32):
    if (type(completed) is not int or type(total) is not int or type(interval) is not int
            or not 1 <= completed <= total or interval < 1):
        raise ValueError('Invalid checkpoint schedule')
    return completed % interval == 0 or completed == total


def validate_domains(domains):
    if not isinstance(domains, dict) or not domains:
        raise ValueError('Domains must be a nonempty mapping')
    for entry in domains.values():
        if not isinstance(entry, (tuple, list)) or len(entry) != 4:
            raise ValueError('Invalid domain entry')
        relation, questions, train, held = entry
        if (not isinstance(relation, str) or not relation
                or not isinstance(questions, (tuple, list)) or len(questions) < 2
                or any(not isinstance(q, str) or '{name}' not in q for q in questions)
                or not isinstance(train, (tuple, list)) or not isinstance(held, (tuple, list))
                or len(train) < 2 or not held
                or any(not isinstance(value, str) or not value for value in (*train, *held))
                or len(set(train)) != len(train) or len(set(held)) != len(held)
                or set(train) & set(held)):
            raise ValueError('Invalid or overlapping domain split')


def add_old_value_replay(cases, domains, old_count, factor):
    if type(old_count) is not int or old_count < 1 or type(factor) is not int or factor < 1:
        raise ValueError('Invalid replay settings')
    if factor > 1 and any(old_count > len(entry[2]) for entry in domains.values()):
        raise ValueError('Replay count exceeds the supervised split')
    old = []
    for row in cases:
        if row['domain'] not in domains or row['fact'] not in domains[row['domain']][2]:
            raise ValueError('Replay cases must be supervised facts')
        if row['fact'] in domains[row['domain']][2][:old_count]:
            old.append(row)
    if not old:
        raise ValueError('Replay selection is empty')
    return list(cases) + old * (factor - 1)


def compare_recovery_rows(restored, previous, recovery_step, comparison_step):
    fields = ('group', 'person', 'domain', 'fact', 'question')
    def index(rows):
        result = {}
        for row in rows:
            key = tuple(row[field] for field in fields)
            if key in result:
                raise ValueError('Duplicate recovery evaluation case')
            result[key] = row['token_ids']
        return result
    current, earlier = index(restored), index(previous)
    shared = current.keys() & earlier.keys()
    same = sum(current[key] == earlier[key] for key in shared)
    exact = current == earlier
    if recovery_step == comparison_step and not exact:
        raise ValueError('Final checkpoint did not reproduce recorded generations')
    return {'matching_generations': same, 'comparable_cases': len(shared),
            'cases': len(current), 'exact_replay': exact and recovery_step == comparison_step}


def choose_counterfactual(values, actual, cycle):
    if (not isinstance(values, (tuple, list)) or len(values) < 2 or len(set(values)) != len(values)
            or actual not in values or type(cycle) is not int or cycle < 0):
        raise ValueError('Invalid counterfactual schedule')
    return values[(values.index(actual) + 1 + cycle % (len(values)-1)) % len(values)]


def counterfactual_loss(own_ce, swapped_ce, margin=0.5):
    """Favor a given full answer under its matching memory over a swapped one."""
    return torch.nn.functional.softplus(margin + own_ce - swapped_ce)


def aligned_teacher_kl(student_logits, teacher_logits, temperature):
    if (not isinstance(student_logits, torch.Tensor) or not isinstance(teacher_logits, torch.Tensor)
            or student_logits.shape != teacher_logits.shape or student_logits.ndim != 2
            or isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('Invalid teacher logits')
    teacher_prob = torch.nn.functional.softmax(teacher_logits.detach().float() / temperature, dim=-1)
    student_log = torch.nn.functional.log_softmax(student_logits.float() / temperature, dim=-1)
    return torch.nn.functional.kl_div(student_log, teacher_prob, reduction='batchmean') * temperature ** 2


def optimizer_groups(params, memory_module, projection_lr, gain_lr):
    if projection_lr is None and gain_lr is None:
        return [{'params': params}]
    if (memory_module is None or projection_lr is None or gain_lr is None
            or any(isinstance(lr, bool) or not isinstance(lr, (int, float))
                   or not math.isfinite(lr) or not 0 < lr <= 0.01 for lr in (projection_lr, gain_lr))):
        raise ValueError('Invalid memory attention learning rates')
    projection = list(memory_module.memory_key.parameters()) + list(memory_module.memory_value.parameters())
    memory_ids = {id(p) for p in projection + [memory_module.gain]}
    if len(memory_ids) != len(projection) + 1 or not memory_ids.issubset({id(p) for p in params}):
        raise ValueError('Invalid memory attention parameter groups')
    base = [param for param in params if id(param) not in memory_ids]
    if not base:
        raise ValueError('Original reader parameters are required')
    return [{'params': base}, {'params': projection, 'lr': float(projection_lr)},
            {'params': [memory_module.gain], 'lr': float(gain_lr)}]


def observe(*, decoder, tokenizer, spec, cross_modules, group, name, expected, question):
    global done
    if done or expected != '苹果':
        return
    params = list(decoder.reader.parameters())
    for layer_idx in (18, 20):
        attn = decoder.base.model.layers[layer_idx].self_attn
        for projection in ('q_proj', 'v_proj'):
            adapter = getattr(attn, projection)
            params += [adapter.a, adapter.b, adapter.condition]
    direct = decoder.base.model.layers[20].self_attn.o_proj
    params += list(direct.to_hidden.parameters()) + [direct.gate]
    wrapped_layer = None
    original_attention = None
    if MEMORY_ATTENTION:
        wrapped_layer = decoder.base.model.layers[23]
        original_attention = wrapped_layer.self_attn
        embedding = decoder.base.model.embed_tokens.weight.detach()
        codec = ContinuousTextCodec(embedding.cpu(), memory_width=90, slot_count=5,
                                    seed=29071, eos_token_id=tokenizer.eos_token_id).to('cuda')
        extension = PretrainedMemoryAttention(original_attention, wrapped_layer.input_layernorm,
                                             decoder, codec, embedding, slots=4)
        wrapped_layer.self_attn = extension
        params += list(extension.memory_key.parameters()) + list(extension.memory_value.parameters()) + [extension.gain]
    if len({id(param) for param in params}) != len(params):
        raise ValueError('Duplicate training parameter')
    names = {id(param): name for name, param in decoder.named_parameters()}
    if any(id(param) not in names for param in params):
        raise ValueError('Training parameter is not registered with the decoder')
    saved = [param.detach().clone() for param in params]
    flags = [module.enabled for module in cross_modules]
    bridge = decoder.base.model.layers[18].self_attn.q_proj.bridge
    bridge_forward = bridge.forward
    try:
        for module in cross_modules:
            module.enabled = False
        for param in params:
            param.requires_grad_(True)
        run(decoder, tokenizer, spec, params, [names[id(param)] for param in params])
        done = True
    finally:
        with torch.no_grad():
            for param, old in zip(params, saved):
                param.copy_(old)
                param.requires_grad_(False)
        for module, flag in zip(cross_modules, flags):
            module.enabled = flag
        bridge.forward = bridge_forward
        if wrapped_layer is not None:
            wrapped_layer.self_attn = original_attention


def run(decoder, tokenizer, spec, params, parameter_names):
    torch.manual_seed(83961)
    rng = random.Random(83961)
    validate_domains(DOMAINS)
    embedding = decoder.base.model.embed_tokens.weight.detach()
    codec = ContinuousTextCodec(embedding.cpu(), memory_width=90, slot_count=5,
                                seed=29071, eos_token_id=tokenizer.eos_token_id).to('cuda')
    if BRIDGE_MODE == 'exact':
        bridge = decoder.base.model.layers[18].self_attn.q_proj.bridge
        bridge.forward = types.MethodType(
            lambda _bridge, codes: context_oracle_embeddings(decoder, codec, embedding, codes), bridge)
    elif BRIDGE_MODE != 'learned':
        raise ValueError('Unsupported bridge mode')
    original = spec['memory_state']
    addresses = spec['memory_addresses']
    memories = {}

    def memory_for(person, fact):
        k = (person, fact)
        if k in memories:
            return memories[k]
        identity = tokenizer.encode(person, add_special_tokens=False)[-1]
        fact_ids = tokenizer.encode(fact, add_special_tokens=False)
        if not 1 <= len(fact_ids) <= 3:
            raise ValueError('Fact exceeds the packet capacity')
        with torch.no_grad():
            packet = codec.encode(torch.tensor([identity, *fact_ids], device='cuda')).reshape(1, -1)
            matrices = []
            for matrix, address in zip(original.current, addresses):
                key = torch.nn.functional.normalize(address.float(), dim=-1)
                old = torch.bmm(matrix.float(), key.unsqueeze(-1)).squeeze(-1)
                scale = old.reshape(1, 5, 18)[:, 0].norm(dim=-1, keepdim=True)
                matrices.append(matrix.float() + (packet * scale - old).unsqueeze(-1) * key.unsqueeze(1))
            read = torch.stack([torch.bmm(matrix, torch.nn.functional.normalize(address.float(), dim=-1).unsqueeze(-1)).squeeze(-1)
                                for matrix, address in zip(matrices, addresses)]).mean(0)[0]
            if codec.decode(read)['token_ids'] != [identity, *fact_ids, tokenizer.eos_token_id]:
                raise ValueError('Controlled packet reconstruction failed')
            memories[k] = {**spec, 'memory_state': replace(original, current=matrices)}
        return memories[k]

    def case(person, domain, fact, question_index=0):
        relation, questions, _, _ = DOMAINS[domain]
        question = questions[question_index].format(name=person)
        answer = f'{person}{relation}{fact}。'
        prompt = p.closed_think_inputs(tokenizer, question, 'cuda')
        targets = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        continuation = torch.tensor([targets[:-1]], device='cuda')
        ids = torch.cat((prompt['input_ids'], continuation), dim=1)
        return {'person': person, 'domain': domain, 'fact': fact, 'question': question,
                'answer': answer, 'prompt': prompt, 'targets': targets, 'ids': ids,
                'memory': memory_for(person, fact)}

    train = [case(person, domain, value, q) for person in NAMES
             for domain, (_, _, values, _) in DOMAINS.items() for value in values for q in (0, 1)]
    train = add_old_value_replay(train, DOMAINS, REPLAY_OLD_COUNT, REPLAY_FACTOR)
    checks = []
    for domain, (_, _, values, held) in DOMAINS.items():
        if EVAL_SEEN_COUNT is not None and (type(EVAL_SEEN_COUNT) is not int or not 1 <= EVAL_SEEN_COUNT <= len(values)):
            raise ValueError('Invalid supervised evaluation count')
        selected = values if EVAL_SEEN_COUNT is None else values[:EVAL_SEEN_COUNT]
        checks += [('train_person', case(NAMES[0], domain, value)) for value in selected]
        checks += [('held_person', case(HELD_NAME, domain, value)) for value in selected]
        checks += [('held_value', case(HELD_NAME, domain, value, HELD_VALUE_Q)) for value in held]
        if EVAL_ALTERNATIVE_QUESTION:
            checks += [('held_value_alt_question', case(HELD_NAME, domain, value, 1)) for value in held]
        if EVAL_NOVEL_QUESTION:
            if len(DOMAINS[domain][1]) < 3:
                raise ValueError('Novel-question evaluation requires a third template')
            checks += [('novel_question_seen', case(HELD_NAME, domain, value, 2)) for value in values]
            checks += [('novel_question_held', case(HELD_NAME, domain, value, 2)) for value in held]

    def generate(item, active=True):
        memory = dict(item['memory']) if active else {}
        memory.pop('use_cache', None)
        with torch.inference_mode():
            ids = decoder.generate(**item['prompt'], **memory, use_cache=True, max_new_tokens=24,
                                   do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id)
        tail = ids[0, item['prompt']['input_ids'].shape[1]:].tolist()
        answer = tokenizer.decode(tail, skip_special_tokens=True).strip()
        correct = answer in (item['answer'], item['answer'][:-1], item['fact'], item['fact']+'。')
        return {'answer': answer, 'token_ids': tail, 'strict_correct': correct}

    def evaluate():
        return [{'group': group, 'person': item['person'], 'domain': item['domain'],
                 'fact': item['fact'], 'question': item['question'], **generate(item)}
                for group, item in checks]

    def load_parameters(directory):
        payload = torch.load(directory/'recovery.pt', map_location='cpu', weights_only=True)
        saved_params = payload.get('params', ())
        if type(payload.get('step')) is not int or not 0 < payload['step'] <= STEPS or len(saved_params) != len(params):
            raise ValueError('Recovery parameters or training step do not match')
        if payload.get('parameter_names', parameter_names) != parameter_names:
            raise ValueError('Recovery parameter names do not match')
        if payload.get('source_sha256', p.sha(audit.SOURCE/'candidate.pt')) != p.sha(audit.SOURCE/'candidate.pt'):
            raise ValueError('Recovery source candidate mismatch')
        if payload.get('base_sha256', p.WEIGHT_SHA) != p.WEIGHT_SHA:
            raise ValueError('Recovery base mismatch')
        if payload.get('bridge_mode', 'learned') != BRIDGE_MODE:
            raise ValueError('Recovery bridge mode mismatch')
        if payload.get('memory_attention', False) != MEMORY_ATTENTION:
            raise ValueError('Recovery memory attention mode mismatch')
        if payload.get('teacher_kl_weight', 0.0) != TEACHER_KL_WEIGHT:
            raise ValueError('Recovery teacher weight mismatch')
        if RELOAD_FROM is not None and INIT_FROM is not None and directory == RELOAD_FROM:
            if payload.get('initial_sha256') != p.sha(INIT_FROM/'recovery.pt'):
                raise ValueError('Recovery initial checkpoint mismatch')
        with torch.no_grad():
            for param, saved in zip(params, saved_params):
                if param.shape != saved.shape:
                    raise ValueError('Recovery parameter dimensions do not match')
                param.copy_(saved.to(param.device))
        return payload

    if RELOAD_FROM is not None:
        payload = load_parameters(RELOAD_FROM)
        restored = evaluate()
        previous = json.loads((RELOAD_FROM/'multirelation-training-result.json').read_text(encoding='utf-8'))['after']
        original_groups = {'train_person', 'held_person', 'held_value', 'held_value_alt_question'}
        comparable_rows = [row for row in restored if row['group'] in original_groups]
        if CROSS_EVALUATION:
            fields = ('group', 'person', 'domain', 'fact', 'question')
            prior_keys = {tuple(row[field] for field in fields) for row in previous}
            comparable_rows = [row for row in comparable_rows if tuple(row[field] for field in fields) in prior_keys]
            current_keys = {tuple(row[field] for field in fields) for row in comparable_rows}
            previous = [row for row in previous if tuple(row[field] for field in fields) in current_keys]
            if not comparable_rows:
                raise ValueError('Cross evaluation has no shared reference cases')
        comparison = compare_recovery_rows(comparable_rows, previous, payload['step'], STEPS)
        ablation = None
        if EVAL_ATTENTION_ABLATION:
            if not MEMORY_ATTENTION:
                raise ValueError('Attention ablation requires the memory attention branch')
            module = decoder.base.model.layers[23].self_attn
            gain = float(module.gain.detach())
            with disable_memory_attention(module):
                disabled = evaluate()
            ablation = {'gain': gain, 'rows': disabled,
                        'changed_generations': sum(a['token_ids'] != b['token_ids']
                                                   for a, b in zip(restored, disabled))}
        (audit.OUT/'reload-result.json').write_text(json.dumps(
            {'recovery_step': payload['step'], 'comparison_step': STEPS,
             **comparison, 'total_evaluated_cases': len(restored),
             'cross_evaluation': CROSS_EVALUATION, 'rows': restored, 'attention_ablation': ablation},
            ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'recovery_step': payload['step'], 'comparison_step': STEPS,
                          **comparison}), flush=True)
        return

    initial_sha = None
    if INIT_FROM is not None:
        initial_sha = p.sha(INIT_FROM/'recovery.pt')
        load_parameters(INIT_FROM)
    before = evaluate()
    if EXPECTED_INITIAL_RESULT is not None:
        reference = json.loads(EXPECTED_INITIAL_RESULT.read_text(encoding='utf-8'))['rows']
        compare_recovery_rows(before, reference, STEPS, STEPS)
    off_checks = [next(item for _, item in checks if item['domain'] == domain) for domain in DOMAINS]
    off_before = [generate(item, False) for item in off_checks]
    memory_module = decoder.base.model.layers[23].self_attn if MEMORY_ATTENTION else None
    groups = optimizer_groups(params, memory_module, MEMORY_PROJECTION_LR, MEMORY_GAIN_LR)
    optimizer = torch.optim.AdamW(groups, lr=2e-5, weight_decay=0, foreach=False)
    trace = []
    visits = {'old': 0, 'new': 0}
    for step in range(STEPS):
        if step % len(train) == 0:
            rng.shuffle(train)
        item = train[step % len(train)]
        subset = DOMAINS[item['domain']][2][:REPLAY_OLD_COUNT]
        visits['old' if item['fact'] in subset else 'new'] += 1
        optimizer.zero_grad(set_to_none=True)
        logits = decoder(input_ids=item['ids'], attention_mask=torch.ones_like(item['ids']),
                         **item['memory'], logits_to_keep=len(item['targets'])).logits[0].float()
        own_ce = torch.nn.functional.cross_entropy(logits, torch.tensor(item['targets'], device='cuda'))
        teacher_kl = torch.zeros((), device='cuda')
        if TEACHER_KL_WEIGHT:
            relation = DOMAINS[item['domain']][0]
            teacher_question = (f'以下是虚构记录：{item["person"]}{relation}{item["fact"]}。\n'
                                f'{item["question"]}')
            teacher_prompt = p.closed_think_inputs(tokenizer, teacher_question, 'cuda')
            continuation = torch.tensor([item['targets'][:-1]], device='cuda')
            teacher_ids = torch.cat((teacher_prompt['input_ids'], continuation), dim=1)
            with torch.no_grad():
                teacher_full = decoder(input_ids=teacher_ids,
                    attention_mask=torch.ones_like(teacher_ids), logits_to_keep=len(item['targets'])).logits[0].float()
            teacher_kl = aligned_teacher_kl(logits, teacher_full,
                                            TEACHER_KL_TEMPERATURE)
        pair_loss = torch.zeros((), device='cuda')
        if COUNTERFACTUAL_WEIGHT:
            values = DOMAINS[item['domain']][2]
            other = choose_counterfactual(values, item['fact'], step // len(train))
            swapped_memory = memory_for(item['person'], other)
            swapped_logits = decoder(input_ids=item['ids'], attention_mask=torch.ones_like(item['ids']),
                                     **swapped_memory, logits_to_keep=len(item['targets'])).logits[0].float()
            swapped_ce = torch.nn.functional.cross_entropy(swapped_logits, torch.tensor(item['targets'], device='cuda'))
            pair_loss = counterfactual_loss(own_ce, swapped_ce)
        loss = own_ce + COUNTERFACTUAL_WEIGHT * pair_loss + TEACHER_KL_WEIGHT * teacher_kl
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        if (step+1) % 16 == 0:
            trace.append({'step': step+1, 'loss': float(loss.detach()), 'own_ce': float(own_ce.detach()),
                          'counterfactual_loss': float(pair_loss.detach()), 'teacher_kl': float(teacher_kl.detach()),
                          'gradient_l2': float(norm)})
            print(json.dumps(trace[-1]), flush=True)
        if checkpoint_due(step + 1, STEPS):
            temp, path = audit.OUT/'recovery.tmp', audit.OUT/'recovery.pt'
            with temp.open('wb') as stream:
                torch.save({'step': step+1, 'params': [param.detach().cpu() for param in params],
                            'parameter_names': parameter_names,
                            'source_sha256': p.sha(audit.SOURCE/'candidate.pt'),
                            'base_sha256': p.WEIGHT_SHA,
                            'counterfactual_weight': COUNTERFACTUAL_WEIGHT,
                            'teacher_kl_weight': TEACHER_KL_WEIGHT,
                            'teacher_kl_temperature': TEACHER_KL_TEMPERATURE,
                            'bridge_mode': BRIDGE_MODE,
                            'memory_attention': MEMORY_ATTENTION,
                            'memory_projection_lr': MEMORY_PROJECTION_LR, 'memory_gain_lr': MEMORY_GAIN_LR,
                            'initial_sha256': initial_sha, 'replay_factor': REPLAY_FACTOR,
                            'replay_old_count': REPLAY_OLD_COUNT, 'visits': dict(visits),
                            'schedule': [(row['person'], row['domain'], row['fact'], row['question'])
                                         for row in train],
                            'optimizer': optimizer.state_dict(), 'rng': rng.getstate()}, stream)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, path)
    after = evaluate()
    off_after = [generate(item, False) for item in off_checks]
    if off_before != off_after:
        raise ValueError('Memory-off ordinary control changed')
    def summary(rows):
        return {group: {'correct': sum(row['strict_correct'] for row in rows if row['group'] == group),
                        'total': sum(row['group'] == group for row in rows)}
                for group in dict.fromkeys(row['group'] for row in rows)}
    result = {'before': before, 'after': after, 'before_summary': summary(before),
              'after_summary': summary(after), 'off_tokens_unchanged': True, 'trace': trace,
              'steps': STEPS, 'train_cases': len(train), 'counterfactual_weight': COUNTERFACTUAL_WEIGHT,
              'teacher_kl_weight': TEACHER_KL_WEIGHT, 'teacher_kl_temperature': TEACHER_KL_TEMPERATURE,
              'bridge_mode': BRIDGE_MODE,
              'memory_attention': MEMORY_ATTENTION,
              'memory_projection_lr': MEMORY_PROJECTION_LR, 'memory_gain_lr': MEMORY_GAIN_LR,
              'initial_sha256': initial_sha, 'replay_factor': REPLAY_FACTOR,
              'replay_old_count': REPLAY_OLD_COUNT, 'visits': visits,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Each test has one supplied memory; the same numeric address is reused across people.',
                              'No automatic relation or entity routing, concurrency or real user state.',
                              'Held values and held person are tested together; their effects are confounded.',
                              'All values were used in previous development diagnostics, not a fresh final test.',
                              'Strict answer scoring can miss correct paraphrases.',
                              'No matched-compute no-training or alternate-seed control.']}
    (audit.OUT/'multirelation-training-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'before': result['before_summary'], 'after': result['after_summary']}), flush=True)


audit.PROBE_OBSERVER = observe
if __name__ == '__main__':
    audit.main()
    if not done:
        raise ValueError('Multirelation training did not run')
