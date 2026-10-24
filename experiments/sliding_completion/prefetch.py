from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'kernel.cu').read_text()
a=s.index('  float a0=0,a1=0,a2=0,a3=0;');b=s.index('  float inv=vscale/denoms[head]',a)
s=s[:a]+r'''
  float gathered[32][4];
  #pragma unroll
  for(int s=0;s<32;s++) {
   float x0=0,x1=0,x2=0,x3=0;
   long at=(((long)(b*HKV+h)*S+s)*8+head)*64+dim;
   if(s<S)asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(x0),"=f"(x1),"=f"(x2),"=f"(x3):"l"(pa+at):"memory");
   gathered[s][0]=x0;gathered[s][1]=x1;gathered[s][2]=x2;gathered[s][3]=x3;
  }
  float a0=0,a1=0,a2=0,a3=0;
  #pragma unroll
  for(int s=0;s<32;s++) {
   float w=weights[head][s];
   a0=fmaf(w,gathered[s][0],a0);a1=fmaf(w,gathered[s][1],a1);a2=fmaf(w,gathered[s][2],a2);a3=fmaf(w,gathered[s][3],a3);
  }
'''+s[b:]
(p/'prefetched.cu').write_text(s)
s=(p/'native.py').read_text().replace("name='muse_completion_v1'","name='muse_completion_prefetched_v1'").replace("ROOT/'kernel.cu'","ROOT/'prefetched.cu'")
(p/'prefetched.py').write_text(s)
for file in ['compare','validate','fresh','sanitize']:
 s=(p/(file+'.py')).read_text().replace('import native','import prefetched as native')
 s=s.replace("results/comparison.json","results/comparison-prefetched.json")
 (p/(file+'_prefetched.py')).write_text(s)
