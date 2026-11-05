"""Bounded synthetic system-identification training from observed transitions."""
import argparse,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.world_time import TimeWorld,WorldTimeConfig


def sample(generator,batch):
    state=(torch.rand(batch,3,generator=generator)*2-1)*torch.tensor([3.,2.,2.])
    action=torch.rand(batch,2,generator=generator)*2-1
    environment=torch.rand(batch,2,generator=generator)*.65+.15
    return state,action,environment


def environment_step(state,action,environment,dt):
    """Analytic synthetic environment. Never imported by model/runtime modules."""
    dt=dt.reshape(-1)
    x,v,temp=state.unbind(-1);force,heater=action.unbind(-1);drag,cooling=environment.unbind(-1)
    decay=torch.exp(-drag*dt);equilibrium=force/drag
    velocity=equilibrium+(v-equilibrium)*decay
    position=x+equilibrium*dt+(v-equilibrium)*(-torch.expm1(-drag*dt))/drag
    heat_decay=torch.exp(-cooling*dt);steady=heater/cooling
    temperature=steady+(temp-steady)*heat_decay
    return torch.stack([position,velocity,temperature],-1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',required=True);a=parser.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(a.seed)
    plan=json.loads((ROOT/'configs/p11-time-plan.json').read_text())
    model=TimeWorld(WorldTimeConfig(hidden=plan['hidden']))
    optimizer=torch.optim.Adam(model.parameters(),lr=plan['learning_rate'])
    generator=torch.Generator().manual_seed(plan['training_seed']+a.seed)
    started=time.perf_counter()
    with (output/'metrics.jsonl').open('w') as log:
        for step in range(plan['training_steps']):
            state,action,env=sample(generator,plan['batch'])
            dt=torch.rand(plan['batch'],1,generator=generator)*.48+.02
            target=environment_step(state,action,env,dt)
            prediction=model.midpoint(state,action,env,dt)
            world_loss=((prediction-target)/dt).square().mean()
            period=torch.rand(plan['batch'],generator=generator)*1.8+.2
            observed=period[:,None]*(1+(torch.rand(plan['batch'],2,generator=generator)*2-1)*.02)
            period_loss=((model.period(observed)-period)/period).square().mean()
            loss=world_loss+period_loss
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            if (step+1)%200==0:
                row={'step':step+1,'world_loss':float(world_loss.detach()),'period_loss':float(period_loss.detach()),
                     'seconds':time.perf_counter()-started}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    model.eval();gen=torch.Generator().manual_seed(plan['development_seed'])
    state,action,env=sample(gen,1024);dt=torch.rand(1024,generator=gen)*1.98+.02
    with torch.no_grad():
        expected=environment_step(state,action,env,dt)
        actual=model.predict(state,action,env,dt)
        rmse=(actual-expected).square().mean(0).sqrt().tolist()
        period=torch.rand(1024,generator=gen)*1.8+.2
        observed=period[:,None]*(1+(torch.rand(1024,2,generator=gen)*2-1)*.02)
        error=float(((model.period(observed)-period).abs()/period).mean())
    report={'seed':a.seed,'steps':plan['training_steps'],'transitions':plan['training_steps']*plan['batch'],
            'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-started,
            'development_rmse':rmse,'development_period_relative_mae':error,'language_tokens':0}
    torch.save({'format':'spora-time-world-v1','config':model.config_dict(),'state':model.state_dict(),'training':report},
               output/'model.pt')
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
