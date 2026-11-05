"""Fresh-process temporal inference: only key, mode, time; no answers or history."""
import argparse
import json
import sys
import torch
from .checkpoint import load_checkpoint
from .temporal_cli import require_temporal


def validate_request(request):
    if not isinstance(request,dict) or set(request)!= {'keys','modes','times'}:
        raise ValueError('Only keys, modes, and times are accepted')
    n=len(request['keys']) if isinstance(request['keys'],list) else 0
    if not 1<=n<=256:raise ValueError('Provide 1 to 256 queries')
    if any(not isinstance(v,list) or len(v)!=n or any(type(x) is not int for x in v) for v in request.values()):
        raise ValueError('Query fields must be matching integer lists')
    return [torch.tensor(request[name],dtype=torch.long).reshape(1,-1) for name in ('keys','modes','times')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--reset-fast',action='store_true');args=p.parse_args()
    torch.set_num_threads(2)
    model,fast,_,_=load_checkpoint(args.checkpoint,'cpu',args.reset_fast)
    require_temporal(model,fast)
    # Python auditing is an I/O check after trusted code/model loading, not an OS sandbox.
    def audit(event,arguments):
        if event in ('open','os.listdir','os.scandir','subprocess.Popen','os.system') or event.startswith('socket.'):
            raise PermissionError('Inference I/O blocked: '+event)
    sys.addaudithook(audit)
    for line in sys.stdin:
        keys,modes,times=validate_request(json.loads(line))
        with torch.no_grad():logits=model.predict_many(keys,fast,modes,times)
        print(json.dumps({'predictions':logits.argmax(-1).flatten().tolist(),
                          'probabilities':logits.softmax(-1).squeeze(0).tolist()}),flush=True)


if __name__=='__main__':main()
