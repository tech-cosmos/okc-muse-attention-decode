from pathlib import Path
s=Path('experiments/sliding/verify_base.py').read_text()
s=s.replace('torch::Tensor qkv, torch::Tensor kc','torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc').replace('torch::Tensor pa, torch::Tensor pm,','torch::Tensor pa, torch::Tensor pm, torch::Tensor out,')
s=s.replace('#include <mma.h>','#include <mma.h>\n#include <cooperative_groups.h>')
s=s.replace('template<int STAGES, int PAGE>','template<int STAGES, int PAGE, int SP>')
s=s.replace('const __nv_bfloat16* __restrict__ qkv, unsigned char* kc','const __nv_bfloat16* __restrict__ qkv, const float* cs, unsigned char* kc')
s=s.replace('float* __restrict__ pm,\n int HQ','float* __restrict__ pm, __nv_bfloat16* out,\n int HQ')
s=s.replace(' __shared__ __align__(16) unsigned char raw[STAGES][2][N*(D+16)];',''' __shared__ __align__(16) union Scratch {
   unsigned char raw[STAGES][2][N*(D+16)];
   struct {float a[64*64];float ml[64*2];} parts;
 } scratch;
 auto& raw=scratch.raw;
 auto cluster=cooperative_groups::this_cluster();
 cluster.sync();''')
s=s.replace(' const int chunk=((seq+N*SPLITS-1)/(N*SPLITS))*N;\n const int lo=split*chunk, hi=min(seq,lo+chunk);',' const int base=max(0,seq-2047);\n const int chunk=((seq-base+N*SP-1)/(N*SP))*N;\n const int lo=base+split*chunk, hi=min(seq,lo+chunk);')
s=s.replace('   Q[row*(D+16)+lane]=to8(x*r); Q[row*(D+16)+lane+32]=to8(y*r);','''   float xp=__bfloat162float(qkv[tok*width+head*D+(lane^1)]);
   float yp=__bfloat162float(qkv[tok*width+head*D+32+(lane^1)]);
   int pos=seq+row/G; float sign=(lane&1)?1.f:-1.f;
   x=x*cs[(long)pos*64+lane/2]+sign*xp*cs[(long)pos*64+32+lane/2];
   y=y*cs[(long)pos*64+16+lane/2]+sign*yp*cs[(long)pos*64+48+lane/2];
   Q[row*(D+16)+lane]=to8(x*r); Q[row*(D+16)+lane+32]=to8(y*r);''')
s=s.replace('   unsigned char k0=to8(x*r), k1=to8(y*r);','''   float xp=__bfloat162float(qkv[tok*width+(HQ+h)*D+(lane^1)]);
   float yp=__bfloat162float(qkv[tok*width+(HQ+h)*D+32+(lane^1)]);
   int pos=seq+row;float sign=(lane&1)?1.f:-1.f;
   x=x*cs[(long)pos*64+lane/2]+sign*xp*cs[(long)pos*64+32+lane/2];
   y=y*cs[(long)pos*64+16+lane/2]+sign*yp*cs[(long)pos*64+48+lane/2];
   unsigned char k0=to8(x*r), k1=to8(y*r);''')
s=s.replace('   return;\n }\n wmma::fragment',' } else {\n wmma::fragment')
s=s.replace('if(isnew || lo+(tile+1)*N>hi)','if(isnew || tile==0 || lo+(tile+1)*N>hi)')
s=s.replace('if(col>bounds[r]) score[n].x[i]=-1e30f;','if(col>bounds[r] || (!isnew && lo+tile*N+col<seq+int(warp)*2+r-2047)) score[n].x[i]=-1e30f;')
a=s.index(' long base=((long)(b*HKV+h)*SPLITS+split)*M;');z=s.index('\nvoid attention(',a)
s=s[:a]+''' consumer_barrier();
 #pragma unroll
 for(int n=0;n<4;n++) wmma::store_matrix_sync(scratch.parts.a+warp*16*D+n*16,acc[n],D,wmma::mem_row_major);
 if(lane%4==0) for(int r=0;r<2;r++) {
   int row=warp*16+lane/4+r*8;
   scratch.parts.ml[row*2]=mx[r];scratch.parts.ml[row*2+1]=sum[r];
 }
 } // consumers
 __syncthreads();
 cluster.sync();
 for(int row=split*5+warp;row<64;row+=SP*5) {
   float mm[SP],ll[SP],xx[SP],yy[SP];float m=-1e30f;
   #pragma unroll
   for(int s=0;s<SP;s++) {
     const float* a=cluster.map_shared_rank(scratch.parts.a,s);
     const float* ml=cluster.map_shared_rank(scratch.parts.ml,s);
     mm[s]=ml[row*2];ll[s]=ml[row*2+1];xx[s]=a[row*64+lane];yy[s]=a[row*64+lane+32];
     m=fmaxf(m,mm[s]);
   }
   float den=0,x=0,y=0;
   #pragma unroll
   for(int s=0;s<SP;s++){float e=exp2f(mm[s]-m);den+=e*ll[s];x+=e*xx[s];y+=e*yy[s];}
   x=x/den*vscale;y=y/den*vscale;float r=rsqrtf(warp_sum(x*x+y*y)/64.f+1e-5f);
   long tok=b*8+row/8,col=(h*8+row%8)*64+lane;
   const __nv_bfloat16* gate=qkv+tok*width+(HQ+2*HKV)*64+col;
   out[tok*HQ*64+col]=__float2bfloat16(x*r/(1.f+expf(-__bfloat162float(gate[0]))));
   out[tok*HQ*64+col+32]=__float2bfloat16(y*r/(1.f+expf(-__bfloat162float(gate[32]))));
 }
 cluster.sync();
}
''' +s[z:]
s=s.replace('void attention(at::Tensor qkv, at::Tensor kc','void attention(at::Tensor qkv, at::Tensor cs, at::Tensor kc').replace('at::Tensor pa, at::Tensor pm, int splits','at::Tensor pa, at::Tensor pm, at::Tensor out, int splits')
a=s.index(' #define LAUNCH(');z=s.index(' C10_CUDA_KERNEL_LAUNCH_CHECK();',a)
s=s[:a]+''' cudaLaunchConfig_t config={};config.gridDim=grid;config.blockDim=dim3(160);config.stream=stream;
 cudaLaunchAttribute attribute={};attribute.id=cudaLaunchAttributeClusterDimension;
 attribute.val.clusterDim.x=splits;attribute.val.clusterDim.y=1;attribute.val.clusterDim.z=1;
 config.attrs=&attribute;config.numAttrs=1;
 #define LAUNCH(ST,P,S) C10_CUDA_CHECK(cudaLaunchKernelEx(&config,verify<ST,P,S>,(__nv_bfloat16*)qkv.data_ptr(),cs.data_ptr<float>(),(unsigned char*)kc.data_ptr(),(unsigned char*)vc.data_ptr(),pt.data_ptr<int>(),seq.data_ptr<int>(),loc.data_ptr<long>(),ks.data_ptr<float>(),vs.data_ptr<float>(),pa.data_ptr<float>(),pm.data_ptr<float>(),(__nv_bfloat16*)out.data_ptr(),hq,hkv,int(pt.size(1)),splits,float(scale)))
 #define SS(ST,P) if(splits==1){LAUNCH(ST,P,1);}else if(splits==2){LAUNCH(ST,P,2);}else if(splits==4){LAUNCH(ST,P,4);}else{LAUNCH(ST,P,8);}
 #define PP(ST) if(page==1){SS(ST,1);}else{SS(ST,64);}
 if(stages==2){PP(2);}else{PP(3);}
''' +s[z:]
s=s.replace("'.build'","'.build_verify'").replace("name='codex_verify_exp'","name='codex_sliding_verify'")
s=s.replace('and not c[\'window\']','and c[\'window\']==2048')
s=s.replace('s=splits or max(1,528//(b*h))','s=splits or 4\n    assert s in (1,2,4,8) and stages in (2,3)')
s=s.replace("module.attention(qkv,data['k_cache']","module.attention(qkv,submission._cos_sin(c['rope_theta'],64,device),data['k_cache']")
s=s.replace('data[\'k_scale\'],data[\'v_scale\'],pa,pm,s,','data[\'k_scale\'],data[\'v_scale\'],pa,pm,out,s,')
a=s.index('    submission._combine[');s=s[:a]+'    return out\n'
Path('experiments/sliding/cuda_verify_cluster.py').write_text(s)
