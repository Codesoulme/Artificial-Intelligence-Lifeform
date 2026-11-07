"""Persistent neural work state and budgeted internal recurrence.

P24 is a finite graph-planning prototype, not natural-language consciousness.
The graph is explicit latest sensory working state, never autobiographical memory.
No search algorithm or stored answer table runs inside this module.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn

SOURCES = {'none': 0, 'user': 1, 'sensor': 2, 'synthetic': 3}
MODES = {'relax': 0, 'focus': 1, 'sleep': 2}


@dataclass
class AwakeConfig:
    nodes: int = 16
    hidden: int = 32
    relax_seconds: float = .25
    focus_seconds: float = .05
    max_gap_seconds: float = 2.
    stable_tolerance: float = .001
    reachable_threshold: float = .03

    def validate(self):
        if not 4 <= self.nodes <= 64 or not 4 <= self.hidden <= 128:
            raise ValueError('Unsupported finite prototype dimensions')
        for value in (self.relax_seconds, self.focus_seconds, self.max_gap_seconds,
                      self.stable_tolerance, self.reachable_threshold):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Positive finite configuration required')
        if not self.focus_seconds <= self.relax_seconds < self.max_gap_seconds:
            raise ValueError('Invalid temporal budget')


class AwakeCore(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or AwakeConfig(); self.config.validate()
        self.transition = nn.Sequential(nn.Linear(2,self.config.hidden), nn.Tanh(),
                                        nn.Linear(self.config.hidden,1))
        self.budget = nn.Sequential(nn.Linear(3,16), nn.Tanh(), nn.Linear(16,3))
        n=self.config.nodes
        for name,value in dict(graph=torch.zeros(n,n),values=torch.zeros(n),
                               goal=torch.zeros(n)).items(): self.register_buffer(name,value)
        for name,value in dict(mode=0,position=0,target=0,has_goal=0,observation_valid=0,
                source=0,urgent=0,revision=0,planned_revision=-1,iterations=0,stable=0,
                ticks=0,neural_steps=0,observations=0,awake_ns=0,sleep_ns=0,
                gap_ns=0,last_utc_ns=0,readiness=0).items():
            self.register_buffer(name,torch.tensor(value,dtype=torch.int64))

    def config_dict(self): return asdict(self.config)

    def validate(self):
        self.config.validate()
        if self.graph.shape!=(self.config.nodes,self.config.nodes) or self.values.shape!=(self.config.nodes,) or self.goal.shape!=self.values.shape:
            raise ValueError('Invalid work tensor shape')
        if not all(bool(torch.isfinite(v).all()) for v in self.state_dict().values()):
            raise ValueError('Nonfinite awake state')
        if not bool(((self.graph==0)|(self.graph==1)).all()): raise ValueError('Binary graph required')
        if int(self.mode) not in MODES.values() or int(self.source) not in SOURCES.values():
            raise ValueError('Unknown mode/source')
        if not 0<=int(self.position)<self.config.nodes or not 0<=int(self.target)<self.config.nodes:
            raise ValueError('Invalid endpoint')
        for name in ('has_goal','observation_valid','urgent','readiness'):
            if int(getattr(self,name)) not in (0,1): raise ValueError('Invalid flag')
        for name in ('revision','iterations','stable','ticks','neural_steps','observations','awake_ns','sleep_ns','gap_ns','last_utc_ns'):
            if int(getattr(self,name))<0: raise ValueError('Negative counter')

    def propagate(self, graph, values, goal):
        neighbor=(graph*values.unsqueeze(-2)).amax(-1)
        inputs=torch.stack((goal,neighbor),-1)
        return self.transition(inputs).squeeze(-1).clamp(0,1)

    @torch.no_grad()
    def reset_plan(self):
        self.values.zero_(); self.iterations.zero_(); self.stable.zero_(); self.readiness.zero_()
        self.planned_revision.fill_(-1)

    @torch.no_grad()
    def observe(self, graph, position, source, urgent=False):
        if source not in SOURCES or source=='none' or type(urgent) is not bool:
            raise ValueError('Explicit input source and boolean urgency required')
        if type(position) is not int or not 0<=position<self.config.nodes:
            raise ValueError('Position outside domain')
        graph=torch.as_tensor(graph,dtype=self.graph.dtype,device=self.graph.device)
        if graph.shape!=self.graph.shape or not bool(((graph==0)|(graph==1)).all()):
            raise ValueError('Binary adjacency required')
        self.graph.copy_(graph);self.position.fill_(position);self.source.fill_(SOURCES[source])
        self.observation_valid.fill_(1);self.observations.add_(1);self.revision.add_(1)
        self.urgent.fill_(int(urgent));self.reset_plan()

    @torch.no_grad()
    def set_goal(self, target):
        if type(target) is not int or not 0<=target<self.config.nodes:
            raise ValueError('Target outside domain')
        self.target.fill_(target);self.goal.zero_();self.goal[target]=1
        self.has_goal.fill_(1);self.reset_plan()

    @torch.no_grad()
    def set_mode(self, mode):
        if mode not in MODES: raise ValueError('Unknown awake mode')
        was_sleep=int(self.mode)==2;self.mode.fill_(MODES[mode])
        if was_sleep and mode!='sleep':
            # Re-observe before acting on potentially stale surroundings.
            self.observation_valid.zero_();self.reset_plan()

    @torch.no_grad()
    def invalidate_gap(self, nanoseconds):
        self.gap_ns.add_(nanoseconds);self.observation_valid.zero_();self.reset_plan()

    @torch.no_grad()
    def tick(self, elapsed_ns, policy='adaptive'):
        if type(elapsed_ns) is not int or elapsed_ns<0: raise ValueError('Nonnegative elapsed time required')
        if policy not in ('adaptive','fixed'): raise ValueError('Unknown budget policy')
        if elapsed_ns>int(self.config.max_gap_seconds*1e9):
            self.invalidate_gap(elapsed_ns);return self.status()
        if int(self.mode)==2:
            self.sleep_ns.add_(elapsed_ns);return self.status()
        self.awake_ns.add_(elapsed_ns);self.ticks.add_(1)
        pending=int(self.has_goal) and int(self.observation_valid) and not int(self.readiness)
        features=self.values.new_tensor([float(int(self.mode)==1),float(pending),float(self.urgent)])
        category=int(self.budget(features).argmax())
        steps=(1,4,8)[category] if policy=='adaptive' else 8
        # Each awake tick really executes a learned transition, even when idle.
        # Idle recurrence is not claimed to be useful new reasoning.
        for _ in range(steps):
            if int(self.observation_valid) and int(self.has_goal):
                updated=self.propagate(self.graph,self.values,self.goal)
                delta=float((updated-self.values).abs().max())
                self.values.copy_(updated);self.iterations.add_(1)
                self.stable.add_(1) if delta<self.config.stable_tolerance else self.stable.zero_()
                if int(self.stable)>=2 and int(self.iterations)>=2:
                    self.readiness.fill_(1);self.planned_revision.copy_(self.revision)
            else:
                self.values.copy_(self.transition(torch.stack((torch.zeros_like(self.values),self.values),-1)).squeeze(-1).clamp(0,1))
            self.neural_steps.add_(1)
        self.urgent.zero_()
        return self.status()

    @torch.no_grad()
    def status(self):
        ready=bool(self.readiness) and bool(self.observation_valid) and bool(self.has_goal) and int(self.mode)!=2 and int(self.planned_revision)==int(self.revision)
        action=None;outcome='needs_observation' if not bool(self.observation_valid) else 'idle'
        if bool(self.has_goal) and bool(self.observation_valid): outcome='planning'
        if ready:
            if int(self.position)==int(self.target):outcome='at_goal'
            elif float(self.values[int(self.position)])<self.config.reachable_threshold:outcome='unreachable_estimate'
            else:
                scores=self.values.masked_fill(self.graph[int(self.position)]==0,-torch.inf)
                candidate=int(scores.argmax())
                if bool(torch.isfinite(scores[candidate])) and float(scores[candidate])>float(self.values[int(self.position)]):
                    action=candidate;outcome='proposal'
                else:outcome='unresolved'
        if int(self.mode)==2:outcome='sleep'
        return dict(mode=next(k for k,v in MODES.items() if v==int(self.mode)),outcome=outcome,
            proposed_node=action,position=int(self.position),target=int(self.target) if bool(self.has_goal) else None,
            ticks=int(self.ticks),neural_steps=int(self.neural_steps),iterations=int(self.iterations),
            observations=int(self.observations),awake_seconds=int(self.awake_ns)/1e9,
            sleep_seconds=int(self.sleep_ns)/1e9,gap_seconds=int(self.gap_ns)/1e9,
            input_source=next(k for k,v in SOURCES.items() if v==int(self.source)))

    def interval(self):
        return self.config.focus_seconds if int(self.mode)==1 or bool(self.urgent) else self.config.relax_seconds


def attach_awake(model, core):
    if hasattr(model,'awake_core'):raise ValueError('Awake extension already attached')
    core.validate();model.awake_core=core.to(next(model.parameters()).device).eval()
    return model
