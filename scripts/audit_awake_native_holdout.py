"""Compare frozen layer-18 candidates on previously unused evaluation cases."""

import argparse
import time
from pathlib import Path

from audit_awake_block_components import fit_norm
from audit_awake_multiturn_layers import ceval_questions, conversation, score_ceval, score_memory
from audit_awake_native_block import INDEX
from audit_awake_native_interpolation import count_switches, paired_states
from deepseek_output_aligned_memory import MODEL, SOURCE, read, sha, torch, write
from spora.pretrained_attention import AbsoluteAttention
from spora.structure_transfer import load_native_block
from train_awake_partial_layer import SPORAFeedForward, TRAIN_FILES, collect, corpus, relative_mse
from train_awake_replacement_replay import load_layer_pair, switch


SUBJECTS = ('high_school_geography', 'middle_school_history', 'middle_school_physics', 'modern_chinese_history')
CASES = (
    ('零件放在左边第二个抽屉，标签放在右边第一个抽屉。标签在哪边？只答左边或右边。', '右边|右',
     '把两样东西交换位置。标签现在在哪边？', '左边|左'),
    ('把十六颗珠子平均分到四个盒里，每盒几颗？只答数字。', '4|四',
     '拿走两个盒子，剩下的盒子一共几颗？只答数字。', '8|八'),
    ('车票原来买在周二，后来改到周四。现在是周几？只答星期。', '周四|星期四',
     '如果往后推一天呢？只答星期。', '周五|星期五'),
    ('只写出“明天见”的最后一个字。', '见',
     '现在改写成第一个字，只写一个字。', '明'),
    ('把九、二、六从大到小排序。用逗号分隔，只写数字。', '9,6,2|九,六,二',
     '然后去掉最小的数，按相同格式回复。', '9,6|九,六'),
    ('储物柜里或许有伞。我还没打开柜门。你能确定里面有伞吗？只答能或不能。', '不能',
     '我打开后看到一把伞。现在能确定了吗？只答能或不能。', '能'),
    ('阿松带的是蓝杯，小卓带的是白杯。谁带了白杯？只写名字。', '小卓',
     '再告诉我另一个人的名字。只写名字。', '阿松'),
    ('请写出“月”的英文，指天上的月亮。只输出一个英文单词。', '(?i:moon)',
     '把刚才那个英文单词的首字母改为大写，只输出结果。', 'Moon'),
)
MEMORY = ((17, '时雨', 1, 6), (18, '徐岩', 2, 5), (19, '关晓', 7, 0),
          (20, '尹宁', 4, 3), (21, '叶川', 6, 1), (22, '白雁', 5, 2))


def layer_components(original, native):
    attention = AbsoluteAttention(original.self_attn, native.max_length)
    attention.qkv = native.block.qkv
    attention.out = native.block.out
    attention.position = native.position
    attention.eval().requires_grad_(False)
    norms = []
    for norm, reference in ((native.block.norm1, original.input_layernorm),
                            (native.block.norm2, original.post_attention_layernorm)):
        fitted = torch.nn.LayerNorm(norm.normalized_shape, eps=norm.eps).to(reference.weight)
        fitted.load_state_dict(norm.state_dict())
        norms.append(fitted.eval().requires_grad_(False))
    feedforward = SPORAFeedForward(native.block.ff).eval().requires_grad_(False)
    return dict(input_layernorm=norms[0], post_attention_layernorm=norms[1],
                self_attn=attention, mlp=feedforward)


def main(output, native_path, candidate_path, ceval, unmodified_control, single_layer_controls, layer12_candidate,
         combine_at_18=False, layer12_components=False, fit_layer12_norms=False):
    base_weights = MODEL / 'model.safetensors'
    paths = (SOURCE, base_weights, MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json',
             native_path, candidate_path, *(ceval / f'{subject}.parquet' for subject in SUBJECTS))
    if layer12_candidate is not None:
        paths = (*paths, layer12_candidate)
    if fit_layer12_norms:
        paths = (*paths, *TRAIN_FILES)
    sources = {path: sha(path) for path in paths}
    started = time.perf_counter()
    decoder, tokenizer, original_layers, replacements, organism, _ = load_layer_pair()
    base = decoder.base
    layers = base.model.layers
    variants = {'pretrained_at_18': layers[INDEX]}
    for name, path in (('native', native_path), ('interpolated', candidate_path)):
        digest = read(path.parent / 'result.json')['candidate_sha256']
        variants[name] = load_native_block(path, base, INDEX, checkpoint_digest=sources[SOURCE],
                                           base_digest=sources[base_weights], candidate_digest=digest)
    limit = min(variants[name].max_length for name in ('native', 'interpolated'))
    questions = list(ceval_questions(ceval, tokenizer, SUBJECTS))
    first, second = paired_states(organism, MEMORY)
    results = {}
    norm_errors = {}
    conditions = {'unmodified': variants['pretrained_at_18']} if unmodified_control else variants
    if single_layer_controls:
        layer18 = variants['native' if combine_at_18 else 'pretrained_at_18']
        conditions = {f'native_at_{index}': layer18 for index in replacements}
    if layer12_candidate is not None:
        digest = read(layer12_candidate.parent / 'result.json')['candidate_sha256']
        calibrated = load_native_block(layer12_candidate, base, 12, checkpoint_digest=sources[SOURCE],
                                       base_digest=sources[base_weights], candidate_digest=digest)
        limit = min(limit, calibrated.max_length)
        layer18 = variants['native' if combine_at_18 else 'pretrained_at_18']
        conditions = {name: layer18 for name in ('baseline_at_12', 'calibrated_at_12')}
        if layer12_components or fit_layer12_norms:
            original = original_layers[12]
            components = layer_components(original, calibrated)
            originals = {key: getattr(original, key) for key in components}
            swaps = {
                'pretrained_at_12': (),
                'norms_at_12': ('input_layernorm', 'post_attention_layernorm'),
                'attention_at_12': ('self_attn',),
                'feedforward_at_12': ('mlp',),
                'hybrid_at_12': tuple(components),
            }
            if fit_layer12_norms:
                layers[12] = original
                layers[INDEX] = layer18
                texts = corpus()
                for key in ('input_layernorm', 'post_attention_layernorm'):
                    reference = originals[key]
                    features, targets = collect(decoder, tokenizer, texts[8:], reference)
                    held, expected = collect(decoder, tokenizer, texts[:8], reference)
                    fitted = fit_norm(reference, features, targets)
                    norm_errors[key] = {}
                    for name, norm in (('transplanted', components[key]), ('fitted', fitted)):
                        norm_errors[key][name] = relative_mse(lambda x: norm(x.to(norm.weight)).float(),
                                                              held, expected)
                    components[key] = fitted
                    print(key, norm_errors[key], flush=True)
                swaps = {'pretrained_at_12': (), 'fitted_norms_at_12': swaps['norms_at_12']}
            conditions = {name: layer18 for name in (*swaps, 'calibrated_at_12')}
    if unmodified_control:
        switch(decoder, original_layers)
    for name, block in conditions.items():
        if single_layer_controls:
            index = int(name.rsplit('_', 1)[1])
            switch(decoder, {**original_layers, index: replacements[index]})
        if name == 'calibrated_at_12':
            layers[12] = calibrated
        elif layer12_components or fit_layer12_norms:
            for key, module in originals.items():
                setattr(original, key, components[key] if key in swaps[name] else module)
            layers[12] = original
        layers[INDEX] = block
        choices = score_ceval(decoder, questions, limit)
        dialogues = [conversation(decoder, tokenizer, case, limit, decoding=dict(limit=48)) for case in CASES]
        summary = dict(ceval_correct=sum(row['choice'] == row['expected'] for row in choices),
                       ceval_total=sum(row['choice'] is not None for row in choices),
                       dialogue_correct=sum(turn['correct'] for group in dialogues for turn in group),
                       dialogue_both=sum(all(turn['correct'] for turn in group) for group in dialogues),
                       dialogue_eos=sum(turn['eos'] for group in dialogues for turn in group))
        result = dict(summary=summary, ceval=choices, dialogue=dialogues)
        if not unmodified_control:
            earlier = score_memory(decoder, tokenizer, first)
            shifted = score_memory(decoder, tokenizer, second)
            summary.update(memory_correct=sum(row['correct'] for row in earlier),
                           memory_shifted_correct=sum(row['correct'] for row in shifted),
                           memory_switches=count_switches(earlier, shifted))
            result.update(memory=earlier, alternate_memory=shifted)
        results[name] = result
        print(name, summary, flush=True)
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('A protected input changed during evaluation')
    write(output, dict(results=results, position_limit=limit, dialogue_cases=len(CASES),
                       combine_at_18=combine_at_18, layer12_components=layer12_components,
                       fitted_norm_errors=norm_errors,
                       memory_cases=0 if unmodified_control else len(first),
                       seconds=time.perf_counter()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                       sources={str(path): digest for path, digest in sources.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument('--unmodified-control', action='store_true',
                          help='Score original DeepSeek on language only with the same position limit')
    controls.add_argument('--single-layer-controls', action='store_true',
                          help='Evaluate each existing replacement independently with the original layer 18')
    controls.add_argument('--layer12-candidate', type=Path,
                          help='Compare a calibrated layer 12 to the integrated block with layer 6 fixed')
    parser.add_argument('--combine-at-18', action='store_true',
                        help='Keep the complete SPORA layer 18 installed during the selected comparison')
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument('--layer12-components', action='store_true',
                             help='Isolate the components of the supplied layer 12 inside the pretrained interface')
    diagnostics.add_argument('--fit-layer12-norms', action='store_true',
                             help='Compare LayerNorm fitted on the original layer 12 input activations')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose an unused output path')
    if args.combine_at_18 and not (args.single_layer_controls or args.layer12_candidate is not None):
        parser.error('--combine-at-18 requires --single-layer-controls or --layer12-candidate')
    if (args.layer12_components or args.fit_layer12_norms) and args.layer12_candidate is None:
        parser.error('Layer 12 diagnostics require --layer12-candidate')
    main(args.output, args.native, args.candidate, args.ceval, args.unmodified_control, args.single_layer_controls,
         args.layer12_candidate, args.combine_at_18, args.layer12_components, args.fit_layer12_norms)
