import fcntl
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 import sys
 sys.path.insert(0,'.');sys.path.insert(0,'experiments/full_decode_tile')
 import candidate,reference,torch,triton
 for ctx in (10,63,64,127,128,255,256,1024):
  data=reference.generate_input(batch=1,ctx=ctx,heads='tp8',layer='full',page=1,q_len=1,seed=28103)
  bn=128;s=triton.cdiv(data['page_table'].shape[1],bn)
  pa=torch.empty((1,2,s,8,64),device='cuda');pm=torch.empty((1,2,s,8,2),device='cuda')
  c=data['config'];qkv=data['qkv']
  candidate.extension(bn).attention(qkv,data['k_scale'],data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['cache_loc'],data['k_scale'],data['v_scale'],pa,pm,1,c['sm_scale']*c['qk_scale_factor']*candidate.parent.LOG2E,True)
  torch.cuda.synchronize()
  q=qkv[0,0,:1024].float().reshape(16,64);q=q*torch.rsqrt((q*q).mean(-1,keepdim=True)+1e-5);q=q.to(torch.float8_e4m3fn).float()
  n=min(ctx+1,bn);idx=data['page_table'][0,:n].long()
  k=data['k_cache'][idx,0,0].float();v=data['v_cache'][idx,0,0].float()
  logits=(q[:8]@k.T)*(c['sm_scale']*c['qk_scale_factor']*candidate.parent.LOG2E)
  m=logits.max(-1).values;l=torch.exp2(logits-m[:,None]);a=l.half().float()@v
  print('PART',ctx,'merr',float((m-pm[0,0,0,:,0]).abs().max()),'denerr',float((l.sum(-1)-pm[0,0,0,:,1]).abs().max()),'aerr',float(torch.linalg.vector_norm(a-pa[0,0,0])/torch.linalg.vector_norm(a)),flush=True)
  out=candidate.kernel(data,128);torch.cuda.synchronize();ok,msg=reference.check_implementation(data,reference.collect(data,out));print(ctx,ok,msg,flush=True)
