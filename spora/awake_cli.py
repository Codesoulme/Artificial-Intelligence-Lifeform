"""Interactive CPU awake runtime; numeric graph input, no desktop control."""
import argparse,json,queue,sys,threading,time
from pathlib import Path
import torch
from .growing import load
from .durable import TensorStore
from .awake_session import AwakeSession


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--individual',default='runs/p24-awake/individual-awake.pt')
    p.add_argument('--session')
    p.add_argument('--resume-session')
    p.add_argument('--demo-seconds',type=float,default=0)
    args=p.parse_args()
    if not 0<=args.demo_seconds<=60:raise ValueError('Demo duration0..60 seconds')
    torch.set_num_threads(1)
    if args.resume_session and args.session:raise ValueError('Choose new session or resume')
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    if args.resume_session:
        path=Path(args.resume_session);store=TensorStore(path)
        try:model,state,metadata,_=store.read()
        except BaseException:store.close();raise
    else:
        model,state,metadata=load(args.individual,'cpu')
        path=Path(args.session) if args.session else Path('runs')/('awake-session-'+str(time.time_ns())+'.sqlite')
        store=TensorStore.create(path,model,state,{**metadata,'runtime':'awake CPU session'})
    with store:
        if not hasattr(model,'awake_core'):raise ValueError('An awake-enabled individual is required')
        session=AwakeSession(model);core=model.awake_core;inbox=queue.Queue(maxsize=64)
        print('持续活动研究入口：目前仅接收数字图环境；尚不能理解任意中文目标。',flush=True)
        print('命令：demo / focus / relax / sleep / wake / status / quit。JSON可提交observe与goal。',flush=True)
        print('本次状态文件：'+str(path.resolve()),flush=True)
        if args.demo_seconds:inbox.put('demo')
        else:
            def reader():
                for line in sys.stdin:inbox.put(line.strip())
                inbox.put('quit')
            threading.Thread(target=reader,daemon=True).start()
        start=time.perf_counter();last_save=start;last_outcome=None
        try:
            while not args.demo_seconds or time.perf_counter()-start<args.demo_seconds:
                # No repeated language-model prompting. Timed callbacks advance
                # the persistent neural work state inside the individual's core.
                try:command=inbox.get(timeout=core.interval())
                except queue.Empty:command=None
                session.tick()
                if command:
                    try:
                        if command=='quit':break
                        if command=='demo':
                            graph=torch.zeros_like(core.graph)
                            for i in range(10):graph[i,i+1]=1
                            core.observe(graph,0,'synthetic');core.set_goal(10)
                            print('已给定一次目标：从节点0到10。接下来由内部循环推进规划。',flush=True)
                        elif command in ('focus','relax','sleep'):core.set_mode(command)
                        elif command=='wake':core.set_mode('relax')
                        elif command=='status':print(json.dumps(core.status(),ensure_ascii=False),flush=True)
                        else:
                            event=json.loads(command)
                            if event['type']=='observe':core.observe(event['graph'],event['position'],event['source'],event.get('urgent',False))
                            elif event['type']=='goal':core.set_goal(event['target'])
                            else:raise ValueError('Unknown input event')
                        # External input can wake computation immediately.
                        if command!='status':
                            session.tick()
                            session.checkpoint(store,model,state)
                    except (ValueError,KeyError,TypeError) as exc:print('输入未接受：'+str(exc),flush=True)
                status=core.status();outcome=(status['outcome'],status['proposed_node'])
                if outcome!=last_outcome:
                    print(json.dumps(status,ensure_ascii=False),flush=True);last_outcome=outcome
                now=time.perf_counter()
                if now-last_save>=5:
                    session.checkpoint(store,model,state);last_save=now
        except KeyboardInterrupt:pass
        finally:
            session.checkpoint(store,model,state)
            print(json.dumps(dict(saved=str(path.resolve()),**core.status()),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
