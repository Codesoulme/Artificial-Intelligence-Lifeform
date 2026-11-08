"""Clock and input plumbing; all planning is inside the model's AwakeCore."""
import time
import hashlib


class AwakeSession:
    def __init__(self,model,monotonic_ns=time.perf_counter_ns,utc_ns=time.time_ns,ledger=None):
        self.core=model.awake_core;self.core.validate()
        self.clock=monotonic_ns;self.utc=utc_ns;self.anchor=self.clock();self.ledger=ledger;self._event_counter=0
        saved=int(self.core.last_utc_ns);now=self.utc()
        # Restart never simulates missing thinking or observations.
        if saved>0 or int(self.core.observations)>0:
            self.core.invalidate_gap(max(0,now-saved) if saved>0 else 0)
        self.offline_gap_known=saved>0 and now>=saved

    def tick(self,policy='adaptive'):
        now=self.clock()
        if now<self.anchor:raise ValueError('Monotonic clock moved backwards')
        delta=now-self.anchor;self.anchor=now
        status=self.core.tick(delta,policy)
        self._record('awake.tick', status, source='clock')
        return status

    def _record(self, kind, payload, source='runtime'):
        if self.ledger is None:
            return
        self._event_counter += 1
        self.ledger.append(kind, payload,
                           event_id=f'{kind}-{self._event_counter}',
                           source=source, state_version=int(self.core.revision))

    def observe(self, graph, position, source, urgent=False):
        self.core.observe(graph, position, source, urgent)
        graph_hash = hashlib.sha256(self.core.graph.detach().cpu().numpy().tobytes()).hexdigest()
        self._record('awake.observe', dict(graph_sha256=graph_hash, position=position,
                                           source=source, urgent=urgent), source=source)

    def set_goal(self, target):
        self.core.set_goal(target)
        self._record('awake.goal', dict(target=target))

    def set_mode(self, mode):
        self.core.set_mode(mode)
        self._record('awake.mode', dict(mode=mode))

    def checkpoint(self,store,model,state):
        if model.awake_core is not self.core:raise ValueError('Wrong individual')
        self.core.last_utc_ns.fill_(self.utc())
        return store.commit(model,state,store.revision)
