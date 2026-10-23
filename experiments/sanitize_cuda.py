"""Small native-only cases for Compute Sanitizer. No timing is performed here."""
import sys
sys.path[:0]=['.','experiments']
import torch, reference, cuda_verify
for page,scale in ((1,1),(64,1/64)):
    for ctx in (10,257,1024):
        data=reference.generate_input(seed=19,batch=2,ctx=ctx,heads='tp8',layer='full',page=page,
                                      q_len=8,dist='edges',kv_scale=scale)
        for stages in (3,5,7):
            for _ in range(2):
                cuda_verify.run(data,splits=4,stages=stages)
        torch.cuda.synchronize()
        print('completed',page,scale,ctx,flush=True)
