"""Read-only finite Chinese questions to the same persisted neural individual."""
import argparse
import json
from pathlib import Path
import sys

import torch

from .chinese_query import answer_memory
from .durable import TensorStore
from .growing import load


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--individual',default='runs/p26-memory-acceptance/individual-query.pt')
    p.add_argument('--question')
    p.add_argument('--details',action='store_true')
    args=p.parse_args()
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(2)
    path=Path(args.individual)
    if path.suffix=='.sqlite':
        with TensorStore(path) as store:model,state,_,_=store.read()
    else:model,state,_=load(path)
    if not hasattr(model,'chinese_query'):raise ValueError('This individual has no trained Chinese query module')
    if args.question is None:
        print('孢子·有限记忆问答：可以查询明确编号对象的当前值、历史记录和来源。')
        print('例如：对象12现在的值是多少？／对象12在事件0的值是多少？／对象12在事件0的来源是什么？')
        print('目前尚不能自由聊天。本入口只查询已有经历；输入 /quit 退出。')
    while True:
        if args.question is not None:question=args.question
        else:
            try:question=input('你：').strip()
            except (EOFError,KeyboardInterrupt):break
        if question=='/quit':break
        if not question:
            if args.question is not None:break
            continue
        result=answer_memory(model,state,question)
        print(json.dumps(result,ensure_ascii=False) if args.details else result['text'],flush=True)
        if args.question is not None:break


if __name__=='__main__':main()
