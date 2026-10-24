from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'tile256.cu').read_text().replace('#include <cuda.h>','#include <cuda.h>\n#include <cooperative_groups.h>')
s=s.replace('float* pa,float* pm,int HQ','float* pa,float* pm,__nv_bfloat16* out,int HQ')
s=s.replace(' asm volatile("griddepcontrol.launch_dependents;":::"memory");','')
a=s.index(' if(warp==0) {\n  long outbase=');z=s.index('\nvoid attention(',a)
s=s[:a]+''' // WGMMA is complete before recycling the dynamic tile storage for partials.
 __syncthreads();
 float* partial=(float*)storage;
 if(warp==0) {
  #pragma unroll
  for(int i=0;i<32;i++)if(!(i&2)) {
   int row=lane/4,col=(i/8)*16+(lane%4)*2+(i&1)+(i&4)*2;
   partial[row*66+col]=acc[i];
  }
  if(lane%4==0){partial[(lane/4)*66+64]=mx[0];partial[(lane/4)*66+65]=sum[0];}
 }
 auto cluster=cooperative_groups::this_cluster();
 cluster.sync();
 // Eight CTAs each finalize one of this KV head's eight query heads.
 if(warp==0) {
  int row=cluster.block_rank();
  float m=-1e30f,ms[8],ls[8],xs[8],ys[8];
  #pragma unroll
  for(int k=0;k<8;k++) {
   float* remote=cluster.map_shared_rank(partial,k);
   ms[k]=remote[row*66+64];ls[k]=remote[row*66+65];
   xs[k]=remote[row*66+2*lane];ys[k]=remote[row*66+2*lane+1];
   m=fmaxf(m,ms[k]);
  }
  float x=0,y=0,den=0;
  #pragma unroll
  for(int k=0;k<8;k++) {
   float w=exp2f(ms[k]-m);
   x=fmaf(w,xs[k],x);y=fmaf(w,ys[k],y);den=fmaf(w,ls[k],den);
  }
  x=x/den*vscale;y=y/den*vscale;
  float r=rsqrtf(sum32(x*x+y*y)/64.f+1e-5f);
  int head=h*8+row;
  float2 gate=__bfloat1622float2(*(const __nv_bfloat162*)(qkv+b*width+(HQ+2*HKV)*64+head*64+2*lane));
  x=x*r/(1.f+expf(-gate.x));y=y*r/(1.f+expf(-gate.y));
  *(__nv_bfloat162*)(out+(b*HQ+head)*64+2*lane)=__floats2bfloat162_rn(x,y);
 }
 // No CTA may exit while a peer is still reading its shared memory.
 cluster.sync();
}
'''+s[z:]
s=s.replace('at::Tensor pa,at::Tensor pm,int page','at::Tensor pa,at::Tensor pm,at::Tensor out,int page')
a=s.index(' #define RUN(P,E)');z=s.index('\n if(page==1)',a)
s=s[:a]+''' cudaLaunchConfig_t config={};config.gridDim=grid;config.blockDim=dim3(128);config.dynamicSmemBytes=65536;config.stream=stream;
 cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={8,1,1};config.attrs=&attr;config.numAttrs=1;
 #define RUN(P,E) C10_CUDA_CHECK(cudaFuncSetAttribute(sliding<P,E>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536)); C10_CUDA_CHECK(cudaLaunchKernelEx(&config,sliding<P,E>,(const __nv_bfloat16*)qkv.data_ptr(),cs.data_ptr<float>(),(unsigned char*)kc.data_ptr(),(unsigned char*)vc.data_ptr(),pt.data_ptr<int>(),seq.data_ptr<int>(),loc.data_ptr<long>(),ks.data_ptr<float>(),vs.data_ptr<float>(),pa.data_ptr<float>(),pm.data_ptr<float>(),(__nv_bfloat16*)out.data_ptr(),hq,hkv,(int)pt.size(1),S,(float)scale))
'''+s[z:]
(p/'cluster.cu').write_text(s)
(p/'cluster_binding.cpp').write_text((p/'binding.cpp').read_text().replace('torch::Tensor pm,int page','torch::Tensor pm,torch::Tensor out,int page'))
