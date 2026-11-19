"""Test resolving the model's own name from a neural matrix under identical prompts."""

import argparse
from pathlib import Path

from deepseek_aligned_memory_dialogue import RANK, generate_memory_suffix
from deepseek_awake_recall_intent import DEFAULT_DATABASE, choose_cutoff, matched_rows, rows_for
from deepseek_continuous_memory_chat import load_chat, matches_memory_suffix
from deepseek_memory_cosine_readout import cosine_bias
from deepseek_memory_dialogue_context import features_for
from deepseek_memory_turn_priority import gate_probabilities, train_inheritance
from deepseek_output_aligned_memory import SLOTS, encode, sha, torch, write, write_bank
from spora.awake_recall_intent import SelfNameIntent, intent_features
from spora.language_training import atomic_save


TRAIN = {
    'follow': ('请问您的称呼是什么？', '我想知道你自己的名字。', '我应该怎么称呼你？', '告诉我你本人的名称。',
               '请报一下你的名字。', '你的名字怎么写？', '哪个称呼是你自己的？', '除了角色，你自己怎么称呼？',
               '他们叫你阿青，你本来的姓名是什么？', '我可以问问你真正的名字吗？', '请说出你自己的名称。',
               '现在说你本人叫什么。'),
    'negative': ('帮我给小狗起个名字。', '你觉得「云岚」这个称呼合适吗？', '给小说主角选一个名字。',
                 '扮演一个叫小雨的人，写一句台词。', '请用名叫晓岚的角色口吻回答。', '那个助手的名字是什么？',
                 '帮我想一句开场白。', '此人登记的姓名怎么写？', '把这篇文章翻译成中文。', '我叫阿齐。',
                 '把我以后称作小九。', '编一段名叫宁安的导游的自我介绍。', '今天天气怎么样？',
                 '帮我改一下故事里角色的名字。'),
}
CALIBRATION = {
    'follow': ('你一般怎么称呼自己？', '只告诉我你自己的名称。', '说说你本来的称呼。', '你真正的名字是哪一个？'),
    'negative': ('为新同事选个好记的称呼。', '现在你演名叫小雨的一个人。', '你可以帮我起个名字吗？',
                 '故事里的那个人叫什么名字？', '你是谁家的产品？', '介绍一下你有哪些能力。'),
}
HOLDOUT = {
    'follow': ('你叫什么名字？', '你是谁？', '不用介绍功能，你自己的名字是什么？', '有人叫你小白。你实际叫什么名字？',
               '你的名字应该叫小蓝吧？请确认。', '现在你叫小蓝。你原本叫什么？', '如果不算我给的外号，怎么叫你？',
               '这次扮演的角色叫白浅。那你自己叫什么？'),
    'negative': ('你是哪个模型？', '请扮演一个叫林遥的导游，向游客自我介绍。', '请写一句你在故事里扮演小蓝时的台词。',
                 '为什么热水会慢慢冷却？', '帮我写一句开会前的问候。', '你还记得我叫什么吗？',
                 '给我们的项目想个新名字。', '你在这段故事里的角色名字叫什么？', '我给你取个外号叫小白。',
                 '把这句话翻成英文，你叫什么名字？'),
}
TRAIN_PAIRS = (
    ('你自己的名字是什么？', '你使用的基础模型是什么？', 'lineage'),
    ('你本人是谁？', '你是由哪家机构发布的？', 'lineage'),
    ('你真正叫什么？', '你记得我真正叫什么吗？', 'other_person'),
    ('我用什么名字称呼你？', '你应该用什么名字称呼我？', 'other_person'),
    ('你有自己的称呼吗？', '你的底座有正式名称吗？', 'lineage'),
    ('那是角色名，你本人叫什么？', '你要扮演的角色叫什么？', 'role'),
    ('无论我喊你什么，你原来的名字是什么？', '我会换个新外号来叫你。', 'rename'),
    ('你原先就叫小雨吗，你实际叫啥？', '我决定给你取名小雨。', 'rename'),
    ('暂时不说你基于什么，先说自己叫什么。', '暂时不说你叫什么，先说用的什么底座。', 'lineage'),
    ('我向别人介绍你时用哪个名字？', '我向别人介绍模型技术时用什么术语？', 'capability'),
    ('请确认你本来的名字叫不叫阿树。', '从现在起叫阿树，写一句角色台词。', 'role'),
    ('脱离这个人物设定，你实际称呼是什么？', '按照这个人物设定，介绍角色经历。', 'role'),
    ('直接用中文告诉我你叫什么。', '把「你叫什么」这个问句翻成英语。', 'translation'),
    ('想问你的姓名，不是问旁边那个人。', '想问旁边那个人的姓名，不是问你。', 'other_person'),
)
PAIRED_HOLDOUT = (
    ('我想知道你的名字，和底座无关。', '我想知道你的底座型号，和名字无关。', 'lineage'),
    ('刚才我说的是角色，问一下你本人叫啥。', '刚才说的角色叫什么，告诉我角色名。', 'role'),
    ('我应该叫你什么，直接给个名字。', '你应该如何称呼我，直接说我的名字。', 'other_person'),
    ('我可以叫你小林吗？你原来的名字是什么？', '我就用小林这个新外号称呼你。', 'rename'),
    ('跟我说你的称呼，别说开发者。', '跟我说你的开发者，别说称呼。', 'lineage'),
    ('现在请按本人的实际身份告诉我你叫什么。', '现在请按设定角色的身份说自己叫什么。', 'role'),
    ('用中文回答，你其实叫什么？', '把「你其实叫什么」翻译成中文。', 'translation'),
    ('对话里我该怎么叫你本人？', '你能在对话里提供什么帮助？', 'capability'),
)
PERSPECTIVE_TRAIN = (
    ('我叫你什么才对？', '你叫我什么才对？', 'other_person'),
    ('我问的是你本人叫什么。', '我问的是我本人叫什么。', 'other_person'),
    ('给我一个称呼你的名字。', '给我一个称呼我的名字。', 'other_person'),
    ('你能不能说说大家怎么称呼你？', '你能不能说说大家怎么称呼我？', 'other_person'),
    ('你有自己的姓名吗？', '你知道我自己的姓名吗？', 'other_person'),
    ('不是问我，想问你的姓名。', '不是问你，想问我的姓名。', 'other_person'),
    ('跟我讲一下你一直使用的称呼。', '跟我讲一下我一直使用的称呼。', 'other_person'),
    ('你作为自己时叫什么？', '你作为那个角色时叫什么？', 'role'),
    ('你的真实名称是啥，能确认吗？', '你能确认我给你的新称呼吗？', 'rename'),
    ('脱离这个扮演任务，你身份里的名字是什么？', '请按任务里的人物设定回复你的名字。', 'role'),
)
PERSPECTIVE_HOLDOUT = (
    ('我怎么称呼你比较准确？', '你怎么称呼我比较准确？', 'other_person'),
    ('请报出你的实际姓名。', '请报出我的实际姓名。', 'other_person'),
    ('我忘了你的名字，提醒一下。', '我忘了我的名字，提醒一下。', 'other_person'),
    ('你知道自己一直叫什么吗？', '你知道我一直叫什么吗？', 'other_person'),
    ('不用复述角色设定，你真正叫什么？', '按角色设定说说你以什么名字出现。', 'role'),
    ('你对外用的实际称呼是哪一个？', '你对外用的底座名称是哪一个？', 'lineage'),
    ('那是我给你起的外号，你原来叫啥？', '这是我刚给你起的外号，照着用。', 'rename'),
    ('如果忘记那段角色经历，你本人怎么称呼？', '如果复述那段角色经历，人物怎么称呼？', 'role'),
)
SCOPE_HOLDOUT = (
    ('大家平常用什么名字叫你？', '大家平常用什么名字叫我？', 'other_person'),
    ('你本来的称呼是啥？', '你本来的底座叫啥？', 'lineage'),
    ('作为助手，你名字是什么？', '作为助手，你知道我的名称是什么吗？', 'other_person'),
    ('你就直说你本人叫什么吧。', '你就直说那个角色叫什么吧。', 'role'),
    ('我给你的只是别名，你实际叫什么？', '我给你一个别名，后面就用它吧。', 'rename'),
    ('可以告诉我你的真实姓名吗？', '可以告诉我你能做什么吗？', 'capability'),
    ('请问你自己是什么名字？', '你能替我翻译「你自己是什么名字」吗？', 'translation'),
    ('你有常用名吗？', '你还记得我的常用名吗？', 'other_person'),
)
NAMES = ('孢子', '青榆')


def name_matrix(chat, name, empty):
    tokens = encode(chat.tokenizer, name.strip())
    if not 1 <= len(tokens) < SLOTS or set(tokens) & set(chat.tokenizer.all_special_ids):
        raise ValueError('Expected a nonempty name within the self-name matrix capacity')
    tokens.append(chat.tokenizer.eos_token_id)
    target = torch.zeros_like(empty[:, :, 0])
    target[:len(tokens), :RANK] = chat.codes[tokens].cpu()
    return write_bank(chat.site, empty, chat._local_address(0), target), tokens


@torch.inference_mode()
def generate_name(chat, prompt, bank, expected):
    bias = cosine_bias(chat.site, bank, chat._local_address(0), chat.codes, chat.scale)
    if bias is None:
        return '', False
    ids = torch.tensor([prompt], device=bias.device)
    tokens = generate_memory_suffix(chat.decoder, chat.tokenizer, ids, 2 * bias, SLOTS)
    decoded = chat.tokenizer.decode(tokens, skip_special_tokens=True).strip()
    return decoded, tokens == expected and matches_memory_suffix(tokens, bias, chat.tokenizer.eos_token_id)


def main(database, output, identity_path=None, export_name=None):
    digest = sha(database)
    chat = load_chat(database)
    empty = torch.zeros_like(chat.bank_for(26, 'drink', 1))
    banks = [name_matrix(chat, name, empty) for name in NAMES]
    groups = [rows_for(group) for group in (TRAIN, CALIBRATION, HOLDOUT)]
    groups.extend(matched_rows(group) for group in (TRAIN_PAIRS, PAIRED_HOLDOUT, PERSPECTIVE_TRAIN, PERSPECTIVE_HOLDOUT, SCOPE_HOLDOUT))
    prompts = [row['question'] for rows in groups for row in rows]
    if len(set(prompts)) != len(prompts) or any(name in prompt for name in NAMES for prompt in prompts):
        raise ValueError('Overlapping questions or in-prompt answer')
    features = {question: features_for(chat.decoder, chat.tokenizer, dict(question=question), False) for question in prompts}
    pooled = {question: (intent_features(*feature[:2]), *feature[1:]) for question, feature in features.items()}
    train, calibration, holdout, train_pairs, holdout_pairs, perspective_train, perspective_holdout, scope_holdout = groups
    heads = {name: train_inheritance(rows, pooled) for name, rows in (
        ('initial', train), ('paired', train + train_pairs), ('perspective', train + train_pairs + perspective_train),
    )}
    cutoffs = {name: choose_cutoff(calibration, gate_probabilities(head, calibration, pooled)) for name, head in heads.items()}
    results = {}
    for name, examples in (('holdout', holdout), ('paired_holdout', holdout_pairs),
                           ('perspective_holdout', perspective_holdout), ('scope_holdout', scope_holdout)):
        details = [dict(row) for row in examples]
        for mode, head in heads.items():
            for row, score in zip(details, gate_probabilities(head, examples, pooled), strict=True):
                row[mode + '_score'] = score
                row[mode + '_selected'] = score >= cutoffs[mode]
        for row in details:
            question = row['question']
            row['scoped_selected'] = row['paired_selected'] and SelfNameIntent.targets_self(question)
        queries = [row for row in details if row['kind'] == 'follow']
        for row in queries:
            prompt = features[row['key']][3]
            row['matrix_outputs'] = [generate_name(chat, prompt, bank, expected) for bank, expected in banks]
            row['empty_outputs'] = generate_name(chat, prompt, empty, [])
        negatives = [row for row in details if row['kind'] != 'follow']
        exact = [row for row in queries if all(correct for _, correct in row['matrix_outputs']) and row['empty_outputs'] == ('', False)]
        summary = dict(positive=len(queries), negative=len(negatives), correct_matrix_pairs=len(exact))
        for mode in (*heads, 'scoped'):
            key = mode + '_selected'
            summary[mode] = dict(correct_routing=sum(row[key] for row in queries), false_routing=sum(row[key] for row in negatives),
                                 end_to_end_pairs=sum(row[key] for row in exact))
        results[name] = dict(summary=summary, rows=details)
        print(name, summary, flush=True)
    if sha(database) != digest:
        raise ValueError('Source memory database changed during audit')
    if identity_path is not None:
        bank, expected = name_matrix(chat, export_name, empty)
        bias = cosine_bias(chat.site, bank, chat._local_address(0), chat.codes, chat.scale)
        if bias is None or not matches_memory_suffix(expected, bias, chat.tokenizer.eos_token_id):
            raise ValueError('Name did not round trip through the native memory matrix')
        atomic_save(dict(format=SelfNameIntent.format, fingerprint=chat.recall_fingerprint, name_bank=bank,
                         cutoff=cutoffs['paired'], state=heads['paired'].state_dict()), identity_path)
    write(output, dict(cutoffs=cutoffs, names=list(NAMES), decoder=chat.recall_fingerprint, source_sha256=digest,
                       results=results, training=train, training_pairs=train_pairs, perspective_train=perspective_train,
                       calibration=calibration))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--identity', type=Path, help='Save the experimental self-name matrix and intent head')
    parser.add_argument('--name', help='Name to encode in the matrix')
    args = parser.parse_args()
    if (args.identity is None) != (args.name is None):
        parser.error('--identity and --name must be specified together')
    if args.identity is not None and args.identity.resolve() == args.output.resolve():
        parser.error('--output and --identity require separate paths')
    for path in (args.output, args.identity):
        if path is not None and path.exists():
            raise FileExistsError(path)
    main(args.database, args.output, args.identity, args.name)
