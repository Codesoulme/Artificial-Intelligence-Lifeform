"""Clock and input plumbing; all planning is inside the model's AwakeCore."""
import time


class AwakeSession:
    def __init__(self,model,monotonic_ns=time.perf_counter_ns,utc_ns=time.time_ns):
        self.core=model.awake_core;self.core.validate()
        self.clock=monotonic_ns;self.utc=utc_ns;self.anchor=self.clock()
        saved=int(self.core.last_utc_ns);now=self.utc()
        # Restart never simulates missing thinking or observations.
        if saved>0 or int(self.core.observations)>0:
            self.core.invalidate_gap(max(0,now-saved) if saved>0 else 0)
        self.offline_gap_known=saved>0 and now>=saved

    def tick(self,policy='adaptive'):
        now=self.clock()
        if now<self.anchor:raise ValueError('Monotonic clock moved backwards')
        delta=now-self.anchor;self.anchor=now
        return self.core.tick(delta,policy)

    def checkpoint(self,store,model,state):
        if model.awake_core is not self.core:raise ValueError('Wrong individual')
        self.core.last_utc_ns.fill_(self.utc())
        return store.commit(model,state,store.revision)
