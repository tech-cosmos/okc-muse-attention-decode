"""Generate CUDA-only padded-row elimination from the committed production source."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parent
source=(ROOT.parents[1]/'submission.py').read_text()
tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_SLIDING' for t in n.targets))
kws={k.arg:ast.literal_eval(k.value) for k in node.value.keywords}
base=kws['cuda_sources']
(ROOT/'base.cu').write_text(base)
start=base.index(' float mx[2]=')
end=base.index(' float acc[32]={0};',start)
replacement=''' float mx[2]={-1e30f,-1e30f},sum[2]={0,0};
 unsigned pr[4][4]={{0}};
 // Only warp zero's first eight rows correspond to real query heads.
 // All warps reconverge before issuing collective PV WGMMA instructions.
 if(warp==0) {
  #pragma unroll
  for(int i=0;i<32;i++)if(!(i&2)) {
   int col=(i/8)*16+(lane%4)*2+(i&1)+(i&4)*2;
   score[i]=lo+col<=seq?score[i]:-1e30f;
   mx[0]=fmaxf(mx[0],score[i]);
  }
  mx[0]=fmaxf(mx[0],__shfl_xor_sync(0xffffffff,mx[0],1));
  mx[0]=fmaxf(mx[0],__shfl_xor_sync(0xffffffff,mx[0],2));
  mx[0]=max(-1e30f,mx[0]*scale*kscale);
  #pragma unroll
  for(int i=0;i<32;i++)if(!(i&2)) {
   int col=(i/8)*16+(lane%4)*2+(i&1)+(i&4)*2;
   float p=lo+col<=seq?exp2f(fmaf(score[i],scale*kscale,-mx[0])):0.f;
   score[i]=p;sum[0]+=p;
  }
  sum[0]+=__shfl_xor_sync(0xffffffff,sum[0],1);
  sum[0]+=__shfl_xor_sync(0xffffffff,sum[0],2);
  #pragma unroll
  for(int d=0;d<4;d++) {
   #pragma unroll
   for(int i=0;i<4;i++)if(!(i&1)) {
    half2 p=__floats2half2_rn(score[d*8+2*i],score[d*8+2*i+1]);pr[d][i]=*(unsigned*)&p;
   }
  }
 }
'''
pruned=base[:start]+replacement+base[end:]
(ROOT/'pruned.cu').write_text(pruned)
(ROOT/'binding.txt').write_text(kws['cpp_sources'])
# Control: eliminate each warp's unused second row, without warp-divergent softmax.
half=pruned.replace(' if(warp==0) {\n  #pragma unroll\n  for(int i=0;i<32;i++)if(!(i&2)) {',' {\n  #pragma unroll\n  for(int i=0;i<32;i++)if(!(i&2)) {',1)
(ROOT/'half.cu').write_text(half)
