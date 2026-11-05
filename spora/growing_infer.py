import argparse,json,sys,torch
from .growing import load,query
from .temporal_infer import validate_request

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);args=p.parse_args()
    torch.set_num_threads(2);model,state,_=load(args.checkpoint)
    def audit(event,args):
        if event in ('open','os.listdir','os.scandir','subprocess.Popen','os.system') or event.startswith('socket.'):
            raise PermissionError('Inference I/O blocked')
    sys.addaudithook(audit)
    for line in sys.stdin:
        k,m,t=validate_request(json.loads(line))
        with torch.no_grad():logits=query(model,state,k.flatten(),m.flatten(),t.flatten())
        print(json.dumps({'predictions':logits.argmax(-1).tolist(),'probabilities':logits.softmax(-1).tolist()}),flush=True)

if __name__=='__main__':main()
