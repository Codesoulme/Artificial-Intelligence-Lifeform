"""Random-initialized compact language path reading the preserved individual.

Own BPE vocabulary and neural implementation; no external model weights.
Memory reads are architecturally connected but not claimed semantically aligned.
"""
from dataclasses import dataclass,asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from .sleep_memory import history_site
from .language_training import load_interface

@dataclass
class ConversationConfig:
    vocabulary:int=8192
    width:int=256
    depth:int=8
    heads:int=8
    hidden:int=704
    context:int=512
    memory_layers:tuple=(3,7)
    memory_read:str='legacy'

    def validate(self):
        if any(type(v) is not int or v<=0 for v in (self.vocabulary,self.width,self.depth,self.heads,self.hidden,self.context)):
            raise ValueError('Positive integer dimensions required')
        if self.memory_read not in ('legacy','typed_v1'):raise ValueError('Unknown memory reader version')
        if self.width%self.heads or (self.width//self.heads)%2:raise ValueError('Even rotary head width required')
        if self.depth<1 or self.context<8 or self.vocabulary<8:raise ValueError('Invalid model dimensions')
        if len(self.memory_layers)!=2 or len(set(self.memory_layers))!=2 or any(i<0 or i>=self.depth for i in self.memory_layers):raise ValueError('Two valid memory sites required')

class RMSNorm(nn.Module):
    def __init__(self,width):
        super().__init__();self.weight=nn.Parameter(torch.ones(width))
    def forward(self,x):return x*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6).to(x.dtype)*self.weight

class ConversationBlock(nn.Module):
    def __init__(self,c):
        super().__init__();self.heads=c.heads;self.head_width=c.width//c.heads
        self.norm1=RMSNorm(c.width);self.norm2=RMSNorm(c.width)
        self.qkv=nn.Linear(c.width,3*c.width,bias=False);self.out=nn.Linear(c.width,c.width,bias=False)
        self.up=nn.Linear(c.width,2*c.hidden,bias=False);self.down=nn.Linear(c.hidden,c.width,bias=False)
        inverse=1/(10000**(torch.arange(0,self.head_width,2).float()/self.head_width))
        angles=torch.outer(torch.arange(c.context).float(),inverse)
        self.register_buffer('cos',angles.cos()[None,None],persistent=False)
        self.register_buffer('sin',angles.sin()[None,None],persistent=False)
    def rotate(self,x,offset):
        n=x.shape[-2];cos=self.cos[:,:,offset:offset+n].to(x.dtype);sin=self.sin[:,:,offset:offset+n].to(x.dtype)
        even,odd=x[...,::2],x[...,1::2]
        return torch.stack((even*cos-odd*sin,even*sin+odd*cos),-1).flatten(-2)
    def forward(self,x,past=None,use_past=False):
        b,n,d=x.shape;offset=0 if past is None else past[0].shape[-2]
        q,k,v=self.qkv(self.norm1(x)).reshape(b,n,3,self.heads,self.head_width).permute(2,0,3,1,4).unbind(0)
        q=self.rotate(q,offset);k=self.rotate(k,offset)
        if past is not None:k=torch.cat((past[0],k),-2);v=torch.cat((past[1],v),-2)
        mask=None
        if past is not None and n>1:
            mask=torch.arange(k.shape[-2],device=x.device)[None,:]<=offset+torch.arange(n,device=x.device)[:,None]
        h=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=past is None)
        x=x+self.out(h.transpose(1,2).reshape(b,n,d))
        gate,value=self.up(self.norm2(x)).chunk(2,-1)
        x=x+self.down(F.silu(gate)*value)
        return x,(k,v) if use_past else None

class ConversationBridge(nn.Module):
    def __init__(self,organism,config=None):
        super().__init__();self.organism=organism;self.config=config or ConversationConfig();self.config.validate()
        for p in organism.parameters():p.requires_grad_(False)
        c=self.config;m=organism.config.memory_width
        self.embedding=nn.Embedding(c.vocabulary,c.width)
        self.blocks=nn.ModuleList([ConversationBlock(c) for _ in range(c.depth)])
        self.norm=RMSNorm(c.width)
        self.memory_query=nn.ModuleList([nn.Linear(c.width,m,bias=False) for _ in range(2)])
        self.memory_out=nn.ModuleList([nn.Linear(m*(2 if c.memory_read=='typed_v1' else 1),c.width,bias=False) for _ in range(2)])
        self.memory_gate=nn.Parameter(torch.full((2,),-3.))
        # Only initialize the new path. Never apply recursively to organism.
        for name,module in self.named_modules():
            if name=='organism' or name.startswith('organism.'):continue
            if isinstance(module,(nn.Linear,nn.Embedding)):nn.init.normal_(module.weight,std=.02)
        for block in self.blocks:
            nn.init.normal_(block.out.weight,std=.02/math.sqrt(2*c.depth));nn.init.normal_(block.down.weight,std=.02/math.sqrt(2*c.depth))
        if c.memory_read=='typed_v1':
            from .typed_memory import TypedMemoryReader
            self.typed_memory=TypedMemoryReader(m)
    def trainable_parameters(self):return [p for p in self.parameters() if p.requires_grad]
    def read_memory(self,x,state,site):
        q=F.normalize(self.memory_query[site](x),dim=-1)
        if self.config.memory_read=='typed_v1':
            read=self.typed_memory(q,state,site,self.organism.config.time_count)
            return x+self.memory_gate[site].sigmoid()*self.memory_out[site](read)
        matrices=[state.current[site]]+[history_site(page,site)[0] for page in state.pages]
        values=torch.stack([torch.einsum('bmd,bnd->bnm',matrix.expand(x.shape[0],-1,-1).detach(),q) for matrix in matrices],2)
        weights=(values*q.unsqueeze(2)).sum(-1).softmax(-1)
        read=(values*weights.unsqueeze(-1)).sum(2)
        return x+self.memory_gate[site].sigmoid()*self.memory_out[site](read)
    def forward(self,tokens,state,past=None,use_past=False):
        if tokens.ndim!=2 or tokens.dtype!=torch.long:raise ValueError('Integer tokens expected')
        if past is not None or use_past:
            from .conversation_state import ConversationPast
        if past is not None:
            if not isinstance(past,ConversationPast):raise ValueError('Unbound incremental state; rebuild the prefix')
            past.validate(self,state)
        if use_past:
            binding=ConversationPast([None]*self.config.depth,self,state)
        offset=0 if past is None else past[0][0].shape[-2]
        if not 0<tokens.shape[1]+offset<=self.config.context:raise ValueError('Conversation exceeds context')
        x=self.embedding(tokens);updated=[];site=0
        for i,block in enumerate(self.blocks):
            x,p=block(x,None if past is None else past[i],use_past)
            if i in self.config.memory_layers:x=self.read_memory(x,state,site);site+=1
            if use_past:updated.append(p)
        logits=F.linear(self.norm(x),self.embedding.weight)
        if past is not None:past.validate(self,state)
        if use_past:
            binding.validate(self,state)
            binding[:]=updated
            return logits,binding
        return logits

def restore_conversation(organism,payload,parent_sha256,tokenizer_sha256,device='cpu'):
    if payload.get('format')!='spora-conversation-v1':raise ValueError('Not a conversation checkpoint')
    if payload.get('tokenizer_sha256')!=tokenizer_sha256:raise ValueError('Wrong tokenizer')
    c=dict(payload['config']);c['memory_layers']=tuple(c['memory_layers']);model=ConversationBridge(organism,ConversationConfig(**c))
    load_interface(model,dict(format='spora-byte-interface-v1',organism_sha256=payload['organism_sha256'],state=payload['state']),parent_sha256)
    return model.to(device).eval()

@torch.no_grad()
def generate(model,state,prefix,eos,budget=128,forbidden=()):
    if not prefix or len(prefix)+budget>model.config.context:raise ValueError('Invalid prompt/budget')
    tokens=torch.tensor([prefix],device=next(model.parameters()).device);output=[];past=None;ended=False
    for _ in range(budget):
        logits,past=model(tokens,state,past=past,use_past=True);scores=logits[0,-1].clone()
        if not bool(torch.isfinite(scores).all()):raise ValueError('Nonfinite generation')
        if forbidden:scores[list(forbidden)]=-torch.inf
        token=int(scores.argmax())
        if token==eos:ended=True;break
        output.append(token);tokens=torch.tensor([[token]],device=tokens.device)
    return dict(tokens=output,ended_with_eos=ended)
