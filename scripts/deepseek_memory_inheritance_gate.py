"""Compare matched-data inheritance gates for elliptical follow-ups."""

import argparse
import time
from pathlib import Path

from deepseek_memory_attribute_span import Route, route_action
from deepseek_memory_dialogue_context import features_for, route
from deepseek_memory_inherited_turn import OUT as INHERITED_RUN, previous_features
from deepseek_memory_turn_priority import OUT as PRIORITY_RUN, combine, gate_probabilities, train_inheritance
from deepseek_output_aligned_memory import MODEL, ROOT, SOURCE, fingerprint, load_integrated, read, sha, torch, write

OUT = ROOT / 'runs/deepseek-memory-inheritance-gate'
TRAIN_PAIRS = (
    ('要查的是采访人爱点的热饮。', '嗯，讲一下？', '嗯，讲一下怎样准备采访？', 'drink', 'ordinary'),
    ('这份材料说的是他的工作身份。', '有写下来没？', '有写下来他喝哪种饮料没？', 'work', 'drink'),
    ('我想核实出差记录上的实际终点。', '好啊，报给我吧。', '好啊，帮我写份差旅申请吧。', 'place', 'ordinary'),
    ('我在看那位工匠当时做什么活儿。', '行，那项怎么写的？', '那项先放着，他最后去了哪座城？', 'work', 'place'),
    ('她的笔记里说到了最喜欢喝的东西。', '有详情没？', '有推荐给同事的热饮没？', 'drink', 'ordinary'),
    ('资料追到这人那次实际去的地方了。', '然后，答案呢？', '然后，他当时做什么工作呢？', 'place', 'work'),
    ('我想弄清他的返程时间。', '请直接说吧。', '请直接给我一个订车建议吧。', 'time', 'ordinary'),
    ('这里要确认入场总共用了多少钱。', '具体是哪一笔？', '具体是哪一处场馆？', 'amount', 'place'),
    ('这是问跟他同行的人。', '方便说一声吗？', '方便说一声他是不是有工牌吗？', 'other', 'other'),
    ('我在查商店开门的时刻。', '好嘛，念念。', '好嘛，写一句开业祝词。', 'time', 'ordinary'),
    ('线索是那人最中意的可饮用东西。', '能告诉我不？', '能告诉我他什么时候出发的吗？', 'drink', 'time'),
    ('这一栏要填她靠哪门手艺维生。', '哦，给个准话。', '哦，改问她那次路上花了多少。', 'work', 'amount'),
)
CAL_PAIRS = (
    ('记录显示她到访过一处地方，我想查名称。', '说说呗。', '说说夏天出门要准备什么。', 'place', 'ordinary'),
    ('那人早上点得最多的饮料是这一项。', '请核实一下。', '请核实他当时在哪个岗位。', 'drink', 'work'),
    ('我想核对他付的票价。', '方便念给我吗？', '方便帮我写个请假条吗？', 'amount', 'ordinary'),
)
HELD_PAIRS = (
    ('这是那家店老板最中意的饮料。', '行，读给我听。', '行，查一下老板什么时候到的。', 'drink', 'time'),
    ('现在看张女士那回到底去了哪一处。', '嗯，方便讲讲么？', '嗯，方便讲讲怎样写一封感谢信么？', 'place', 'ordinary'),
    ('登记簿标了他谋生的职业。', '有的话把它告诉我。', '有的话把他喜欢喝的也告诉我。', 'work', 'drink'),
    ('档案里问的是约见发生在几号。', '知道就回我一句。', '知道就回我一句他去的是哪儿。', 'time', 'place'),
    ('这部分要谈聊他最终落脚的去处。', '能报一个么？', '能报一个旅游团的例子么？', 'place', 'ordinary'),
    ('我正在找两位见面时的真正介绍人。', '麻烦读一下原文。', '麻烦读一下原文里的那个岗位名称。', 'other', 'work'),
    ('她喜欢的饮品写在那份笔记里。', '说出来好了。', '说出来怎么给新员工安排培训好了。', 'drink', 'ordinary'),
    ('这条记录是他去现场花的总费用。', '我想再确认下。', '我想再确认下他具体做的是哪种工作。', 'amount', 'work'),
)


def pairs(rows):
    return [dict(key=f'pair-{i}-{turn}', history=h, question=q, field=f, target_turn=turn)
            for i, (h, follow, explicit, inherited, direct) in enumerate(rows)
            for turn, (q, f) in enumerate(((follow, inherited), (explicit, direct)))]


def current_features(decoder, tokenizer, rows):
    return {row['key']: features_for(decoder, tokenizer, row, False) for row in rows}


def pooled(features):
    return {key: (torch.cat((tail, tokens.mean(0))), tokens, labels, ids, turns)
            for key, (tail, tokens, labels, ids, turns) in features.items()}


def choose_threshold(rows, probabilities):
    grid = []
    for cutoff in (.25, .35, .5, .65, .75):
        false_inherit = sum(r['target_turn'] == 1 and p >= cutoff for r, p in zip(rows, probabilities, strict=True))
        missed = sum(r['target_turn'] == 0 and p < cutoff for r, p in zip(rows, probabilities, strict=True))
        grid.append(dict(cutoff=cutoff, false_inherit=false_inherit, missed=missed))
    return min(grid, key=lambda r: (r['false_inherit'] + r['missed'], r['false_inherit'], -r['cutoff']))['cutoff'], grid


def main(out):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    torch.set_num_threads(2)
    priority_paths = [PRIORITY_RUN / name for name in ('plan.json', 'routes.json', 'routing.pt')]
    inherited_paths = [INHERITED_RUN / name for name in ('plan.json', 'previous-router.pt')]
    for run, paths in ((PRIORITY_RUN, priority_paths), (INHERITED_RUN, inherited_paths)):
        digests = read(run / 'result.json')['artifact_sha256']
        if any(sha(path) != digests[path.name] for path in paths):
            raise ValueError('Input artifact changed')
    original_plan, setup = (read(path) for path in priority_paths[:2])
    inherited_plan = read(inherited_paths[0])
    suites = {**inherited_plan['suites'], 'new_pairs': pairs(HELD_PAIRS)}
    calibration = original_plan['calibration'] + [dict(row, key='cal-' + row['key']) for row in pairs(CAL_PAIRS)]
    suites['calibration'] = calibration
    train = original_plan['train'] + [dict(row, target_turn=1) for row in original_plan['direct_train']] + pairs(TRAIN_PAIRS)
    train = [dict(row, key=f'train-{index}', kind='follow' if row['target_turn'] == 0 else 'direct')
             for index, row in enumerate(train)]
    if {r['question'] for r in train} & {r['question'] for rows in suites.values() for r in rows}:
        raise ValueError('Training question occurs in evaluation')
    inputs = (Path(__file__), SOURCE, *priority_paths, *inherited_paths, *(MODEL / name for name in
              ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')))
    hashes = fingerprint(inputs)
    out.mkdir()
    write(out / 'plan.json', dict(train=train, suites=suites, sources=hashes))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
              dtype=torch.float16, attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, organism, state = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    del organism, state
    decoder.eval().requires_grad_(False)
    width = base.lm_head.weight.shape[1]
    priority = torch.load(priority_paths[2], weights_only=True)
    original = torch.nn.Linear(width, 1).eval()
    original.load_state_dict(priority['inheritance'])
    direct, previous = (Route(width, localized=True).eval() for _ in range(2))
    direct.load_state_dict(priority['direct'])
    previous.load_state_dict(torch.load(inherited_paths[1], weights_only=True))
    features = current_features(decoder, tokenizer, train)
    tail = train_inheritance(train, features)
    mean = train_inheritance(train, pooled(features))
    heads = {'original': original, 'tail': tail, 'tail_and_mean': mean}
    path = out / 'gates.pt'
    torch.save({name: head.state_dict() for name, head in heads.items() if name != 'original'}, path)
    for name, state in torch.load(path, weights_only=True).items():
        heads[name].load_state_dict(state)

    grid, results = {}, {}
    cutoffs = {'original': setup['gate_cutoff']}
    for suite, rows in suites.items():
        current = current_features(decoder, tokenizer, rows)
        inherited = route(previous, rows, previous_features(decoder, tokenizer, rows))
        explicit = route(direct, rows, current)
        results[suite] = {}
        for name, head in heads.items():
            features = pooled(current) if name == 'tail_and_mean' else current
            probabilities = gate_probabilities(head, rows, features)
            if suite == 'calibration' and name != 'original':
                cutoffs[name], grid[name] = choose_threshold(rows, probabilities)
            picked = combine(inherited, explicit, probabilities, cutoffs[name])
            results[suite][name] = [dict(row, action=route_action(row, setup['cutoffs']['priority'])) for row in picked]
        print(f'Evaluated {suite}: {len(rows)} conversations', flush=True)
    if fingerprint(inputs) != hashes:
        raise ValueError('Input changed during evaluation')
    write(out / 'result.json', dict(suites=results, cutoffs=cutoffs, calibration_grid=grid,
          elapsed_seconds=time.perf_counter() - started, artifact_sha256={name: sha(out / name) for name in
          ('plan.json', 'gates.pt')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    main(parser.parse_args().out)
