"""Explicit opt-in local conversation candidate; never replaces default chat."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tokenizers import Tokenizer
import torch
from spora.growing import load
from spora.conversation import restore_conversation,generate
from scripts.prepare_conversation import sha
ROOT=Path(__file__).resolve().parents[1]

def prefix_tokens(tokenizer,messages):
    tokens=[tokenizer.token_to_id('<|bos|>')];eos=tokenizer.token_to_id('<|eos|>')
    for role,text in messages:
        tokens.append(tokenizer.token_to_id('<|'+role+'|>'))
        tokens.extend(tokenizer.encode(text.replace('<|','＜｜')).ids);tokens.append(eos)
    return tokens+[tokenizer.token_to_id('<|assistant|>')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--tokenizer',default='runs/conversation-data-v1/tokenizer.json')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--prompt');p.add_argument('--budget',type=int,default=96)
    args=p.parse_args();plan=json.loads((ROOT/'configs/p23-conversation.json').read_text('utf-8'))
    if sha(ROOT/plan['parent'])!=plan['parent_sha256']:raise ValueError('Original individual changed')
    torch.set_num_threads(2);organism,state,_=load(ROOT/plan['parent'],args.device)
    payload=torch.load(ROOT/args.checkpoint,map_location='cpu',weights_only=True)
    tokenizer=Tokenizer.from_file(str(ROOT/args.tokenizer));model=restore_conversation(organism,payload,plan['parent_sha256'],sha(ROOT/args.tokenizer),args.device)
    if not 1<=args.budget<model.config.context:raise ValueError('Invalid generation budget')
    history=[];eos=tokenizer.token_to_id('<|eos|>');forbidden=tuple(tokenizer.token_to_id(x) for x in ('<|pad|>','<|bos|>','<|user|>','<|assistant|>'))
    if args.prompt is None:print('研究候选，尚未通过聊天验收。/quit退出，/clear清空本次对话。不会写入长期经历。')
    while True:
        text=args.prompt if args.prompt is not None else input('你：').strip()
        if text=='/quit':break
        if text=='/clear':history=[];print('本次对话已清空。');continue
        if not text:continue
        history.append(('user',text));dropped=False
        while len(prefix_tokens(tokenizer,history))+args.budget>model.config.context and len(history)>1:
            history=history[2:];dropped=True
        prefix=prefix_tokens(tokenizer,history)
        if len(prefix)+args.budget>model.config.context:
            history.pop();print('这条输入超过当前长度，请缩短后重试。')
            if args.prompt is not None:break
            continue
        started=time.perf_counter();result=generate(model,state,prefix,eos,args.budget,forbidden)
        answer=tokenizer.decode(result['tokens'],skip_special_tokens=False);seconds=time.perf_counter()-started
        if args.prompt is not None:
            print(json.dumps(dict(prompt=text,text=answer,seconds=seconds,**result),ensure_ascii=False));break
        if dropped:print('较早的对话轮次已移出本次输入范围。')
        print('模型：'+answer)
        if not result['ended_with_eos']:print('（已达到本轮输出长度上限。）')
        history.append(('assistant',answer))

if __name__=='__main__':main()
