"""Compare paired conversion partials and new cache bytes with native baseline."""
import fcntl
import sys
sys.path.insert(0, '.')
import torch
import reference
from experiments.sliding_fp8pairs import base, query, both

with open('/tmp/okc-gpu.lock', 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for batch, ctx, page, heads, scale in [
        (1, 10, 1, 'tp8', 1.), (3, 2048, 64, 'tp8', 1./64),
        (8, 80000, 1, 'tp8', 1.), (1, 4000, 64, 'dp', 1./64),
    ]:
        for seed in (103, 711, 900):
            d = reference.generate_input(batch=batch, ctx=ctx, page=page, heads=heads,
                layer='sliding', q_len=1, kv_scale=scale, dist='soft', seed=seed)
            hkv = d['config']['num_kv_heads']
            splits = min(32, (d['page_table'].shape[1] * page + 63)//64)
            inv = 1. / (d['config']['rope_theta'] ** (torch.arange(0,64,2,device='cuda')/64))
            ang = torch.arange(ctx+1,device='cuda')[:,None] * inv
            cs = torch.cat([ang.cos(),ang.sin()],-1).contiguous()
            slots = d['cache_loc']
            answers = []
            for mod in (base, query, both):
                pa = torch.empty((batch,hkv,splits,8,64), device='cuda')
                pm = torch.empty((batch,hkv,splits,8,2), device='cuda')
                mod.module.attention(d['qkv'], cs, d['k_cache'], d['v_cache'], d['page_table'],
                    d['seq_lens'], d['cache_loc'], d['k_scale'], d['v_scale'], pa, pm, page,
                    d['config']['sm_scale']*d['config']['qk_scale_factor']*1.4426950408889634, True)
                torch.cuda.synchronize()
                kb = d['k_cache'].view(torch.uint8).reshape(-1,hkv,64)[slots].clone()
                vb = d['v_cache'].view(torch.uint8).reshape(-1,hkv,64)[slots].clone()
                answers.append((pa,pm,kb,vb))
            for name, answer in zip(('query','both'), answers[1:]):
                assert all(torch.equal(a,b) for a,b in zip(answers[0], answer)), (name,batch,ctx,page,heads,scale,seed)
            print('PASS bit-exact partials/cache', batch,ctx,page,heads,scale,seed,flush=True)
            del d, answers, pa, pm, kb, vb
            torch.cuda.empty_cache()
