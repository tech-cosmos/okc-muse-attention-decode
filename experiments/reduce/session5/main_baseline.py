"""Exact inherited Triton full merge, with all attention and plans unchanged."""
import triton,submission
class MainMerge:
 @staticmethod
 def combine_full(qkv,pa,pm,vs,out,hq,hkv,length,splits,pdl):
  submission._combine[(qkv.shape[0]*length,hq)](qkv,pa,pm,vs,out,HQ=hq,HKV=hkv,G=hq//hkv,L=length,M=length*hq//hkv,D=64,S=splits,S_PAD=triton.next_power_of_2(splits),SB=min(128,triton.next_power_of_2(splits)),PDL=pdl,num_warps=4 if splits>32 else 2,launch_pdl=pdl)
mods={'main':MainMerge}
