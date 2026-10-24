from pathlib import Path
ROOT=Path(__file__).resolve().parent
src=Path('submission.py').read_text().split('_NATIVE_SLIDING = ')[1].split("cuda_sources=r'''")[1].split("'''",1)[0]
# Keep the single-tile arithmetic exactly, replace only completion/epilogue.
src=src.replace('template<int PAGE,bool EARLY>','template<int PAGE,bool EARLY,bool FUSE>')
src=src.replace('float* pa,float* pm,int HQ','float* pa,float* pm,int* counts,__nv_bfloat16* output,int HQ')
src=src.replace('asm volatile("griddepcontrol.launch_dependents;":::"memory");','if(!FUSE) asm volatile("griddepcontrol.launch_dependents;":::"memory");')
end=src.index('\n}\nvoid attention(')
epilogue=r'''
 if(FUSE) {
  // Every partial writer publishes its stores; CTA synchronization hands the
  // publication to the leader, whose device-scoped acq_rel atomic chains CTAs.
  if(warp==0)__threadfence();
  __syncthreads();
  __shared__ int last;
  if(tid==0) {
   unsigned old;
   asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;":"=r"(old):"l"(counts+b*HKV+h):"memory");
   last=old==S-1;
  }
  __syncthreads();
  if(!last)return;
  __shared__ float weights[8][32],denoms[8];
  #pragma unroll
  for(int j=0;j<2;j++) {
   int head=warp*2+j;
   long at=((long)(b*HKV+h)*S+lane)*8+head;
   float m=-1e30f,l=0;
   if(lane<S)asm volatile("ld.global.cg.v2.f32 {%0,%1},[%2];":"=f"(m),"=f"(l):"l"(pm+at*2):"memory");
   float mx=m;
   #pragma unroll
   for(int d=16;d;d/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
   float w=lane<S?exp2f(m-mx):0.f;
   weights[head][lane]=w;
   float denom=sum32(w*l);
   if(lane==0)denoms[head]=denom;
  }
  __syncthreads();
  int head=tid/16,dim=(tid%16)*4;
  float a0=0,a1=0,a2=0,a3=0;
  #pragma unroll
  for(int s=0;s<32;s++)if(s<S) {
   float x0,x1,x2,x3;
   long at=(((long)(b*HKV+h)*S+s)*8+head)*64+dim;
   asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(x0),"=f"(x1),"=f"(x2),"=f"(x3):"l"(pa+at):"memory");
   float w=weights[head][s];
   a0=fmaf(w,x0,a0);a1=fmaf(w,x1,a1);a2=fmaf(w,x2,a2);a3=fmaf(w,x3,a3);
  }
  float inv=vscale/denoms[head];a0*=inv;a1*=inv;a2*=inv;a3*=inv;
  float sq=a0*a0+a1*a1+a2*a2+a3*a3;
  #pragma unroll
  for(int d=8;d;d/=2)sq+=__shfl_xor_sync(0xffffffff,sq,d);
  float norm=rsqrtf(sq/64.f+1e-5f);
  long gi=(long)b*width+(HQ+2*HKV+h*8+head)*64+dim;
  long oi=(long)b*HQ*64+(h*8+head)*64+dim;
  float vals[4]={a0,a1,a2,a3};
  #pragma unroll
  for(int j=0;j<4;j++)output[oi+j]=__float2bfloat16_rn(vals[j]*norm/(1.f+expf(-__bfloat162float(qkv[gi+j]))));
  // No block uses the counter after publishing. Reset for the next call.
  if(tid==0)asm volatile("st.release.gpu.global.u32 [%0],0;"::"l"(counts+b*HKV+h):"memory");
 }
'''
src=src[:end]+epilogue+src[end:]
src=src.replace('at::Tensor pa,at::Tensor pm,int page,double scale,bool early)', 'at::Tensor pa,at::Tensor pm,at::Tensor counts,at::Tensor output,int page,double scale,bool fuse)')
src=src.replace('#define RUN(P,E) sliding<P,E>', '#define RUN(P,F) sliding<P,true,F>')
src=src.replace('pm.data_ptr<float>(),hq,', 'pm.data_ptr<float>(),counts.data_ptr<int>(),( __nv_bfloat16*)output.data_ptr(),hq,')
src=src.replace('if(early)', 'if(fuse)')
(ROOT/'kernel.cu').write_text(src)
