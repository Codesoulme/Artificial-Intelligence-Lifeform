"""Prepare a conservative role-marked multi-turn Chinese dialogue subset."""
from __future__ import annotations
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'runs/conversation-data-v5'
OUTPUT = ROOT/'runs/multiturn-dialogue-v7'
REPORT = ROOT/'reports/P33-multiturn-preparation-v7.json'
TOKENIZER = SOURCE/'tokenizer.json'

BAD = re.compile(r'(作为AI|人工智能助手|人工智能|AI|我的朋友|最近工作|喝咖啡|我今天|我最近|我刚|我嘛|我一直|我有空|我遇到|我曾|我的经历|我通过|我常|我会|我愿|我觉得|我认为|我的目的|存在的目的是|意识|智能|主观体验|最近看|最近读|看过|读了|和朋友|和你聊天|心情|帮了不少|在为|看了几部|最近挺|最近确实|互联网上搜索|实时新闻|股票|投资|汇率|诊断|治疗|药物|诉讼|版权|法律意见|量子计算|伦理|意识上传|数字治理|跨文化|哲学)', re.I)
CONTROL = re.compile(r'(<\||```|https?://|\[图片\]|\ufffd)')
REPEAT = re.compile(r'(.{2,18})\1{2,}')
LIST = re.compile(r'(?:^|\n)\s*(?:\d+[.、)]|[-*•])')
PERSONAL = re.compile(r'(今天我|最近我|我还没|我扶|我帮|我得|我刚|我遇到|我曾|我朋友|和朋友|和你聊天|我想|我读|我看过|我看了|我在学习|我正在)', re.I)
PERSONAL_CONTEXT = re.compile(r'(最近忙|最近挺|最近确实|今天早上|今天心情|今天过得|我的心情|我有一个|我有空|我感觉|我觉得|我认为|我想|我会|我读|我看|我帮|我扶)', re.I)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_reason(messages):
    if not 4 <= len(messages) <= 10 or len(messages) % 2:
        return 'turn_count'
    if [m.get('role') for m in messages] != ['user','assistant']*(len(messages)//2):
        return 'role_sequence'
    for message in messages:
        text = message.get('content', '').strip()
        if not text or len(text) > 360 or CONTROL.search(text) or BAD.search(text):
            return 'unsafe_or_meta_content'
        if message.get('role') == 'assistant' and PERSONAL.search(text):
            return 'invented_personal_experience'
        if message.get('role') == 'assistant' and PERSONAL_CONTEXT.search(text):
            return 'invented_personal_context'
        han = sum('\u3400' <= c <= '\u9fff' for c in text)
        latin = sum('a' <= c.lower() <= 'z' for c in text)
        if han < 4 or han / max(1, han + latin) < .72:
            return 'low_chinese_ratio'
        if REPEAT.search(re.sub(r'\s+', '', text)):
            return 'repeated_phrase'
    answers = [messages[i]['content'] for i in range(1, len(messages), 2)]
    if any(LIST.search(answer) or answer.count('\n') > 2 for answer in answers):
        return 'list_heavy_answer'
    if not all(re.search(r'[。！？!?；;]$', answer) for answer in answers):
        return 'unfinished_answer'
    return None


def main():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError('Preserve existing output')
    manifest = json.loads((SOURCE/'manifest.json').read_text('utf-8'))
    for name in ('dialogue-train.jsonl', 'tokenizer.json'):
        if sha(SOURCE/name) != manifest['files'][name]:
            raise ValueError('Source artifact hash mismatch: '+name)
    stats = collections.Counter(); kept=0; targets=0; rows=[]
    OUTPUT.mkdir(parents=True)
    with (SOURCE/'dialogue-train.jsonl').open(encoding='utf-8') as source, \
         (OUTPUT/'dialogue.jsonl').open('x', encoding='utf-8') as raw:
        for line_no, line in enumerate(source, 1):
            row=json.loads(line); reason=row_reason(row['messages'])
            if reason:
                stats['removed_'+reason]+=1; continue
            raw.write(line); rows.append((line_no, row)); kept+=1
            targets += sum(len(message['content']) for message in row['messages']
                           if message['role'] == 'assistant')
    # Keep raw source records and a deterministic audit sample; token encoding is
    # performed by the next gate after a sample review, never here silently.
    sample_seed=330108
    sample_indices=[int(i) for i in __import__('random').Random(sample_seed).sample(range(len(rows)), min(128,len(rows)))] if rows else []
    result={'status':'prepared_pending_manual_review','source':'runs/conversation-data-v5/dialogue-train.jsonl',
            'source_manifest_sha256':sha(SOURCE/'manifest.json'),'source_file_sha256':sha(SOURCE/'dialogue-train.jsonl'),
            'tokenizer_sha256':sha(TOKENIZER),'rows_kept':kept,'assistant_content_characters':targets,'stats':dict(stats),
            'sample_seed':sample_seed,'sample_count':len(sample_indices),
            'sample_row_numbers_1based':[rows[i][0] for i in sample_indices],
            'files':{'dialogue.jsonl':sha(OUTPUT/'dialogue.jsonl')},
            'scope':'Mechanical role-marked multi-turn candidate only. Filters attempt to reduce unsafe, meta, invented-experience and list-heavy records but do not establish their absence. Requires review before tokenization or training. Characters are not token targets.',
            'ready_for_training':False}
    (OUTPUT/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    REPORT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('rows_kept','sample_count','stats')}))


if __name__=='__main__':main()
