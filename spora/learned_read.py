"""Learned score factors with deterministic residual-error certification.

Factors are distilled from actual resident neural matrices, not observations or
answer tables. They propose pages; only a bound may authorize skipped reads.
"""
from dataclasses import replace
import math,time
import torch
from .memory_read import MemoryReadPlan
from .sleep_memory import history_site


def train_factors(matrices,rank=12,steps=400,seed=2410000):
    if type(rank) is not int or not 1<=rank<=matrices[0].shape[-1] or type(steps) is not int or not 1<=steps<=400:
        raise ValueError('Invalid bounded factor training')
    bank=torch.stack([m[0].detach() for m in matrices])
    target=(bank+bank.transpose(-1,-2))*.5
    generator=torch.Generator(device=bank.device).manual_seed(seed)
    left=(torch.randn(bank.shape[0],bank.shape[1],rank,device=bank.device,generator=generator)*.03).requires_grad_()
    right=(torch.randn(bank.shape[0],rank,bank.shape[2],device=bank.device,generator=generator)*.03).requires_grad_()
    optimizer=torch.optim.Adam([left,right],lr=.01)
    start=time.perf_counter()
    with torch.enable_grad():
        for _ in range(steps):
            loss=((left@right-target).square().mean((-1,-2))/target.square().mean((-1,-2)).clamp_min(1e-8)).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    return [[l.detach().clone(),r.detach().clone()] for l,r in zip(left,right)], {
        'steps':steps,'rank':rank,'seconds':time.perf_counter()-start,'relative_mse':float(loss.detach())}


class LearnedReadPlan(MemoryReadPlan):
    def __init__(self,matrices,factors):
        super().__init__(matrices)
        if len(factors)!=len(matrices):raise ValueError('One score factor pair per memory region required')
        self.factor_sources=tuple(t for pair in factors for t in pair)
        self.factor_versions=tuple(t._version for t in self.factor_sources)
        self.left=torch.stack([p[0].detach().clone() for p in factors])
        self.right=torch.stack([p[1].detach().clone() for p in factors])
        if self.left.ndim!=3 or self.right.ndim!=3 or self.left.shape[1]!=self.width or self.right.shape[1:]!=(self.left.shape[2],self.width):
            raise ValueError('Invalid score factors')
        if self.left.device!=self.banks.device or self.right.device!=self.banks.device or self.left.dtype!=self.banks.dtype or self.right.dtype!=self.banks.dtype:
            raise ValueError('Score factors must match memory device and precision')
        if not bool(torch.isfinite(self.left).all() and torch.isfinite(self.right).all()):raise ValueError('Nonfinite score factors')
        # Quadratic form ignores the antisymmetric part of a matrix.
        bank=self.banks[self.mapping].double()
        residual=(bank+bank.transpose(-1,-2))*.5-self.left.double()@self.right.double()
        epsilon=torch.finfo(self.banks.dtype).eps
        self.radius=(residual.square().sum((-1,-2)).sqrt()*(1+64*epsilon)+64*epsilon).to(self.banks.dtype)
        self.norm=(bank.square().sum((-1,-2)).sqrt()*(1+64*epsilon)+64*epsilon).to(self.banks.dtype)

    def matches_factors(self,factors):
        incoming=tuple(t for pair in factors for t in pair)
        return len(incoming)==len(self.factor_sources) and all(a is b and a._version==version for a,b,version in zip(incoming,self.factor_sources,self.factor_versions))

    @property
    def tensor_bytes(self):
        return super().tensor_bytes+sum(t.numel()*t.element_size() for t in (self.left,self.right,self.radius,self.norm))

    def read(self,queries,page_keys,mode='learned',error_l2=1e-4,max_selected=2):
        if mode!='learned':return super().read(queries,page_keys,mode,error_l2,max_selected)
        if torch.is_grad_enabled():raise RuntimeError('Learned plans are inference-only')
        if not self.matches(self.sources):raise ValueError('Memory changed; rebuild score bounds')
        if not all(t._version==v for t,v in zip(self.factor_sources,self.factor_versions)):raise ValueError('Score factors changed; rebuild bounds')
        if not math.isfinite(error_l2) or error_l2<0 or type(max_selected) is not int or max_selected<1:
            raise ValueError('Invalid bounded read request')
        if queries.ndim<2 or queries.shape[-1]!=self.width or page_keys.shape!=(self.page_count,self.width) or queries.numel()==0:
            raise ValueError('Invalid read dimensions')
        if any(x.device!=self.banks.device or x.dtype!=self.banks.dtype or not bool(torch.isfinite(x).all()) for x in (queries,page_keys)):
            raise ValueError('Finite, matching precision/device inputs required')
        shape=queries.shape;q=queries.reshape(-1,self.width);n=len(q)
        proxy=(torch.einsum('nd,pdr->npr',q,self.left)*torch.einsum('prd,nd->npr',self.right,q)).sum(-1)
        proxy=(proxy+q@page_keys.T)/math.sqrt(self.width)
        normq=q.square().sum(-1).sqrt()
        radius=normq[:,None].square()*self.radius/math.sqrt(self.width)
        roundoff=64*torch.finfo(q.dtype).eps*(proxy.abs()+radius+1)
        lower,upper=proxy-radius-roundoff,proxy+radius+roundoff
        order=proxy.argsort(-1,descending=True)
        lo,up=lower.gather(1,order),upper.gather(1,order)
        counts=torch.zeros(n,dtype=torch.long,device=q.device);bounds=q.new_zeros(n)
        limit=min(max_selected,self.page_count-1,self.banks.shape[0]-1)
        for k in range(1,limit+1):
            mass=torch.sigmoid(torch.logsumexp(up[:,k:],-1)-torch.logsumexp(lo[:,:k],-1))
            bound=2*self.norm.max()*normq*mass
            accept=(counts==0)&torch.isfinite(bound)&(bound<=error_l2)
            counts=torch.where(accept,k,counts);bounds=torch.where(accept,bound,bounds)
        full=counts==0;nf=int(full.sum());result=torch.empty_like(q);operations=nf*self.banks.shape[0]
        if nf:result[full]=self._full(q[full],page_keys)
        for k in range(1,limit+1):
            mask=counts==k;size=int(mask.sum())
            if size:result[mask]=self._selected(q[mask],page_keys,order[mask,:k]);operations+=size*k
        return result.reshape(shape),{'mode':'learned','queries':n,'pages':self.page_count,
            'matrix_vectors':operations,'baseline_matrix_vectors':n*self.page_count,'fallback_queries':nf,
            'pruned_queries':n-nf,'max_read_error_bound':float(bounds.max()),
            'score_multiply_adds':2*n*self.page_count*self.width*self.left.shape[-1]}


def prepare_router(model,state,rank=12,steps=400,seed=2410000):
    if state.internal is None:raise ValueError('Internal control state required')
    factors,reports=[],[]
    for site in range(len(model.memories)):
        matrices=[state.current[site]]+[history_site(p,site)[0] for p in state.pages]
        trained,report=train_factors(matrices,rank,steps,seed+site)
        factors.append(trained);reports.append(report)
    return replace(state,internal=replace(state.internal,router_factors=factors,router_cursor=state.cursor)),reports
