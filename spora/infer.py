"""Stateless inference worker. Receives checkpoint and keys, never targets."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch
from .checkpoint import load_checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--reset-fast',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    model,fast,_,_=load_checkpoint(args.checkpoint,'cpu',args.reset_fast)
    if fast[0][0].shape[0]!=1:
        raise ValueError('Inference checkpoint must contain exactly one individual')
    # Imports and checkpoint loading finish before guard: forward cannot read files/network.
    def audit(event, arguments):
        if event in ('open','os.listdir','os.scandir','subprocess.Popen','os.system') or event.startswith('socket.'):
            raise PermissionError('Inference I/O blocked: '+event)
    sys.addaudithook(audit)
    for line in sys.stdin:
        request=json.loads(line)
        if set(request)!= {'keys'}:
            raise ValueError('Inference accepts only current query keys')
        keys=torch.tensor(request['keys'],dtype=torch.long).reshape(1,-1)
        with torch.no_grad():
            logits=model.predict_many(keys,fast)
        print(json.dumps({'predictions':logits.argmax(-1).flatten().tolist(),
                          'probabilities':logits.softmax(-1).squeeze(0).tolist()}),flush=True)


if __name__=='__main__':
    main()
