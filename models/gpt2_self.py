import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(self.n_embd, 3*self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)

    def forward(self,x):
        B,T,C = x.shape
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1,2).view(B,T,C).contiguous()
        y = self.c_proj(y)
        return y
        
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
        
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        



class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size,config.n_embd),
        wpe = nn.Embedding(config.block_size,config.n_embd),
        h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
        ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight
    
    def forward(self,x, targets=None,use_cache=None,cache=None):
        B,T = x.shape
        emb = self.transformer.wte(x)
        pos = torch.arange(0,T,dtype=torch.long,device=x.device)
        pos = self.transformer.wpe(pos)
        x = emb+pos
        for layer in self.transformer.h:
            x = layer(x)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        return x
        
         
    
    

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        #assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        ## n_layer, n_head and n_embd are determined from model_type
        #config_args = {
        #    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        #    'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        #    'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        #    'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        #}[model_type]
        #config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        #config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        ## create a from-scratch initialized minGPT model
        #config = GPTConfig(**config_args)


        model_hf = GPT2LMHeadModel.from_pretrained(model_type)

        n_layer = model_hf.config.n_layer
        n_head = model_hf.config.n_head
        n_embd = model_hf.config.n_embd
        vocab_size = model_hf.config.vocab_size


        config = GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd, vocab_size=vocab_size)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type,master_process=False):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer