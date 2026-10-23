from pathlib import Path
s=Path('experiments/sliding/cuda_sliding_v1.py').read_text()
s=s.replace('#include <cuda/atomic>','#include <cuda/atomic>\n#include <cooperative_groups.h>')
s=s.replace('template<int STAGES,int W,int PAGE,bool FUSED>','template<int STAGES,int W,int PAGE,int SPLITS>')
s=s.replace(' __shared__ __align__(16) unsigned char raw[W][STAGES][2][N*D];',''' __shared__ __align__(16) union Scratch {
   unsigned char raw[W][STAGES][2][N*D];
   struct {float a[W*8*64];float ml[W*8*2];} parts;
 } scratch;
 auto& raw=scratch.raw;
 auto cluster=cooperative_groups::this_cluster();
 cluster.sync();''')
s=s.replace('const int base=max(0,seq-2047),total=S*W;','const int base=max(0,seq-2047),total=SPLITS*W;')
a=s.index(' long part=((long)(b*HKV+h)*total+ws)*8;')
z=s.index('\nvoid attention(',a)
s=s[:a]+''' __syncthreads();
 float* spa=scratch.parts.a;float* spm=scratch.parts.ml;
 #pragma unroll
 for(int n=0;n<4;n++) {
   #pragma unroll
   for(int i=0;i<8;i++) if(!(i&2)) {
     int row=lane/4,col=n*16+(lane%4)*2+(i&1)+(i&4)*2;
     spa[(warp*8+row)*64+col]=acc[n].x[i];
   }
 }
 if(lane%4==0){spm[(warp*8+lane/4)*2]=mx;spm[(warp*8+lane/4)*2+1]=sum;}
 cluster.sync();
 for(int row=split*W+warp;row<8;row+=SPLITS*W) {
   float mm[SPLITS][W],ll[SPLITS][W],xx[SPLITS][W],yy[SPLITS][W];
   float m=-1e30f;
   #pragma unroll
   for(int s=0;s<SPLITS;s++) {
     const float* a=cluster.map_shared_rank(spa,s);const float* ml=cluster.map_shared_rank(spm,s);
     #pragma unroll
     for(int w=0;w<W;w++) {
       int p=w*8+row;
       mm[s][w]=ml[p*2];ll[s][w]=ml[p*2+1];
       xx[s][w]=a[p*64+lane];yy[s][w]=a[p*64+lane+32];
       m=fmaxf(m,mm[s][w]);
     }
   }
   float den=0,x=0,y=0;
   #pragma unroll
   for(int s=0;s<SPLITS;s++) {
     #pragma unroll
     for(int w=0;w<W;w++) {
       float e=exp2f(mm[s][w]-m);
       den+=e*ll[s][w];x+=e*xx[s][w];y+=e*yy[s][w];
     }
   }
   x=x/den*vscale;y=y/den*vscale;
   float r=rsqrtf(warp_sum(x*x+y*y)/64.f+1e-5f);
   long col=(h*8+row)*64+lane;
   const __nv_bfloat16* gate=qkv+b*width+(HQ+2*HKV)*64+col;
   out[(long)b*HQ*64+col]=__float2bfloat16(x*r/(1.f+expf(-__bfloat162float(gate[0]))));
   out[(long)b*HQ*64+col+32]=__float2bfloat16(y*r/(1.f+expf(-__bfloat162float(gate[32]))));
 }
 cluster.sync();
}
''' + s[z:]
a=s.index(' #define LAUNCH(');z=s.index(' C10_CUDA_KERNEL_LAUNCH_CHECK();',a)
s=s[:a]+''' cudaLaunchConfig_t config={};config.gridDim=grid;config.stream=stream;
 cudaLaunchAttribute attribute={};attribute.id=cudaLaunchAttributeClusterDimension;
 attribute.val.clusterDim.x=splits;attribute.val.clusterDim.y=1;attribute.val.clusterDim.z=1;
 config.attrs=&attribute;config.numAttrs=1;
 #define LAUNCH(ST,W,P,SP) config.blockDim=dim3(W*32); C10_CUDA_CHECK(cudaLaunchKernelEx(&config,sliding<ST,W,P,SP>,(__nv_bfloat16*)qkv.data_ptr(),cs.data_ptr<float>(),(unsigned char*)kc.data_ptr(),(unsigned char*)vc.data_ptr(),pt.data_ptr<int>(),seq.data_ptr<int>(),loc.data_ptr<long>(),ks.data_ptr<float>(),vs.data_ptr<float>(),pa.data_ptr<float>(),pm.data_ptr<float>(),count.data_ptr<int>(),(__nv_bfloat16*)out.data_ptr(),hq,hkv,int(pt.size(1)),splits,float(scale)))
 #define SS(ST,W,P) if(splits==1){LAUNCH(ST,W,P,1);}else if(splits==2){LAUNCH(ST,W,P,2);}else if(splits==4){LAUNCH(ST,W,P,4);}else{LAUNCH(ST,W,P,8);}
 #define PP(ST,W) if(page==1){SS(ST,W,1);}else{SS(ST,W,64);}
 #define WW(ST) if(warps==1){PP(ST,1);}else if(warps==2){PP(ST,2);}else{PP(ST,4);}
 WW(2);
''' + s[z:]
s=s.replace("'.build'","'.build_cluster'").replace("name='codex_sliding_exp'","name='codex_sliding_cluster'")
s=s.replace('assert c[\'q_len\']','assert fused and stages==2 and splits in (1,2,4,8)\n    assert c[\'q_len\']')
s=s.replace('splits=4,warps=4,stages=3','splits=4,warps=4,stages=2')
Path('experiments/sliding/cuda_cluster.py').write_text(s)
