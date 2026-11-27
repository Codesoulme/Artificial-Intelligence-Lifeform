"""Read-only source audit; output aggregates only, never source dialogue."""
import ast, collections, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/research/perltqa-source-20261127'
OUT=ROOT/'runs/research/perltqa-source-20261127/audit.json'

def flatten(value):
    if isinstance(value,str):return [value]
    if isinstance(value,list):return [part for item in value for part in flatten(item)]
    if isinstance(value,dict):return [part for item in value.values() for part in flatten(item)]
    return []

def main():
    if OUT.exists():raise FileExistsError(OUT)
    manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf8'))
    for name,record in manifest['files'].items():
        data=(SOURCE/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Source hash mismatch: '+name)
    base=SOURCE/'Dataset/zh'
    memory=json.loads((base/'perltmem.json').read_text(encoding='utf8'))
    questions=json.loads((base/'perltqa.json').read_text(encoding='utf8'))
    mem_names=[x['profile']['Protagonist'] for x in memory]
    qa_names=[name for item in questions for name in item]
    if len(mem_names)!=len(set(mem_names)) or len(qa_names)!=len(set(qa_names)):raise ValueError('Duplicate name')
    mem={x['profile']['Protagonist']:x for x in memory}
    qa={name:sections for item in questions for name,sections in item.items()}
    counts=collections.Counter();texts=collections.defaultdict(list);lengths=[];shapes=collections.Counter()
    for name,sections in qa.items():
        if name not in mem:counts['missing_character']+=1;continue
        for key,rows in sections['dialogues'].items():
            counts['dialogue_groups']+=1;counts['questions']+=len(rows)
            if key not in mem[name]['dialogues']:
                counts['missing_dialogue']+=1;continue
            contents=mem[name]['dialogues'][key]['contents']
            shapes[type(contents).__name__]+=1
            text='\n'.join(flatten(contents));lengths.append(len(text))
            if not text:counts['empty_dialogue']+=1
            texts[hashlib.sha256(text.encode('utf8')).hexdigest()].append(name)
            for row in rows:
                if not row.get('Question') or not row.get('Answer'):counts['missing_qa']+=1
                if row.get('Memory Anchors'):counts['questions_with_anchor']+=1
                try:
                    refs=ast.literal_eval(row['Reference Memory'])
                    if not isinstance(refs,list):raise ValueError('reference not list')
                    counts['reference_lists']+=1
                    if key in refs:counts['reference_contains_group']+=1
                    if len(refs)>1:counts['multiple_references']+=1
                except (ValueError,SyntaxError,TypeError,KeyError):counts['invalid_reference']+=1
    result={'commit':manifest['commit'],'memory_characters':len(mem_names),'question_characters':len(qa_names),
        'counts':dict(counts),'contents_types':dict(shapes),'unique_exact_dialogue_texts':len(texts),
        'duplicate_groups':sum(len(v)-1 for v in texts.values()),
        'cross_character_repeated_texts':sum(len(set(v))>1 for v in texts.values()),
        'length_chars':{'min':min(lengths),'median':sorted(lengths)[len(lengths)//2],'max':max(lengths)},
        'limitations':['Exact dialogue content comparison only; shared facts, paraphrases, and source ties not checked.',
            'Reference points to a group; no claim that answers are entailed or anchors are correct.',
            'Synthetic dialogues and noncommercial dataset; no independent natural human dialogue claim.']}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
