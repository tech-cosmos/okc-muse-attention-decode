from pathlib import Path
s=Path('experiments/sliding/cuda_merge.py').read_text()
a=s.index(' float2 ml[S];float2 a[S];float m=-1e30f;');z=s.index(' x=x/den*vscale;',a)
s=s[:a]+''' float2 ml={-1e30f,0};
 if(lane<S) {
   const float* mp=(const float*)(pm+base+lane*MG);
   asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(ml.x), "=f"(ml.y) : "l"(mp) : "memory");
 }
 float m=ml.x;
 #pragma unroll
 for(int d=16;d;d/=2)m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,d));
 float weight=lane<S?exp2f(ml.x-m):0.f;
 float den=weight*ml.y;
 #pragma unroll
 for(int d=16;d;d/=2)den+=__shfl_xor_sync(0xffffffff,den,d);
 float2 a[S];
 #pragma unroll
 for(int s=0;s<S;s++) {
   const float* ap=pa+(base+s*MG)*64+2*lane;
   asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(a[s].x), "=f"(a[s].y) : "l"(ap) : "memory");
 }
 float x=0,y=0;
 #pragma unroll
 for(int s=0;s<S;s++) {
   float w=__shfl_sync(0xffffffff,weight,s);
   x=fmaf(w,a[s].x,x);y=fmaf(w,a[s].y,y);
 }
''' +s[z:]
s=s.replace('case 12:{WW(12);break;}','').replace('case 24:{WW(24);break;}','').replace('case 64:{WW(64);break;}','').replace('case 128:{WW(128);break;}','')
s=s.replace("'.build_merge'","'.build_warp_merge'").replace("name='codex_merge_exp'","name='codex_warp_merge'")
Path('experiments/sliding/cuda_merge_warp.py').write_text(s)
