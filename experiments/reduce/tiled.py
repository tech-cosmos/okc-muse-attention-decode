"""Split/dimension tiled native reduction, compared with one-split-row-per-warp loads."""
import native
from torch.utils.cpp_extension import load_inline
prefix=native.CUDA[:native.CUDA.index('template<int W,int N,bool PDL>')]
body=r'''
template<int W,int N,bool PDL>
__global__ __launch_bounds__(W*32) void reduce_kernel(const __nv_bfloat16* qkv,
 const float* pa,const float2* pm,const float* vs,__nv_bfloat16* out,
 int HQ,int HKV,int L,int B,int S) {
 constexpr int T=LANES, SUB=32/T, NR=(N+SUB-1)/SUB, ND=64/T;
 unsigned lane=threadIdx.x%32,warp=threadIdx.x/32,sl=lane/T,dl=lane%T;
 int row=blockIdx.x;
 int tok=row/HQ,head=row%HQ,b=tok/L,g=HQ/HKV,h=head/g;
 int width=(2*HQ+2*HKV)*64,rr=(tok%L)*g+head%g,MG=L*g;
 int base=((b*HKV+h)*S)*MG+rr;
 float2 gate=__bfloat1622float2(*(const __nv_bfloat162*)(qkv+tok*width+(HQ+2*HKV)*64+head*64+2*lane));
 float vscale=*vs;
 if(PDL)asm volatile("griddepcontrol.wait;":::"memory");
 float2 ml[NR];float m=-1e30f;
 #pragma unroll
 for(int i=0;i<NR;i++) {
   int s=warp*N+i*SUB+sl;
   ml[i]=s<S?read2((const float*)(pm+base+s*MG)):make_float2(-1e30f,0.f);
   m=fmaxf(m,ml[i].x);
 }
 #pragma unroll
 for(int d=T;d<32;d*=2)m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,d));
 float wt[NR],den=0;
 #pragma unroll
 for(int i=0;i<NR;i++) {wt[i]=exp2f(ml[i].x-m);den=fmaf(wt[i],ml[i].y,den);}
 #pragma unroll
 for(int d=T;d<32;d*=2)den+=__shfl_xor_sync(0xffffffff,den,d);
 float a[NR][ND];
 #pragma unroll
 for(int i=0;i<NR;i++) {
   int s=warp*N+i*SUB+sl;
   #pragma unroll
   for(int d=0;d<ND;d+=4) {
     if(s<S) {
       const float* p=pa+(base+s*MG)*64+d*T+dl*4;
       asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(a[i][d]),"=f"(a[i][d+1]),"=f"(a[i][d+2]),"=f"(a[i][d+3]):"l"(p):"memory");
     } else {a[i][d]=a[i][d+1]=a[i][d+2]=a[i][d+3]=0;}
   }
 }
 float x[ND]={};
 #pragma unroll
 for(int i=0;i<NR;i++) {
   #pragma unroll
   for(int d=0;d<ND;d++)x[d]=fmaf(wt[i],a[i][d],x[d]);
 }
 #pragma unroll
 for(int j=T;j<32;j*=2){
   #pragma unroll
   for(int d=0;d<ND;d++)x[d]+=__shfl_xor_sync(0xffffffff,x[d],j);
 }
 __shared__ float partial[W][68];
 if(sl==0) {
   #pragma unroll
   for(int d=0;d<ND;d++)partial[warp][(d/4)*T*4+dl*4+d%4]=x[d];
 }
 if(lane==0){partial[warp][64]=m;partial[warp][65]=den;}
 __syncthreads();
 if(warp!=0)return;
 float gm=-1e30f;
 #pragma unroll
 for(int w=0;w<W;w++)gm=fmaxf(gm,partial[w][64]);
 float ox=0,oy=0;den=0;
 #pragma unroll
 for(int w=0;w<W;w++) {
   float sc=exp2f(partial[w][64]-gm);
   ox=fmaf(sc,partial[w][2*lane],ox);oy=fmaf(sc,partial[w][2*lane+1],oy);
   den=fmaf(sc,partial[w][65],den);
 }
 ox=ox/den*vscale;oy=oy/den*vscale;
 float sq=sum32(ox*ox+oy*oy),r=rsqrtf(sq/64.f+1e-5f);
 ox=ox*r/(1.f+expf(-gate.x));oy=oy*r/(1.f+expf(-gate.y));
 *(__nv_bfloat162*)(out+row*64+2*lane)=__floats2bfloat162_rn(ox,oy);
}
'''
host=native.CUDA[native.CUDA.index('void reduce(at::Tensor'):]
MODS={}
def get(lanes):
    if lanes not in MODS:
        MODS[lanes]=load_inline(name=f'muse_reduce_tiled{lanes}',cpp_sources='''
#include <torch/extension.h>
void reduce(torch::Tensor qkv,torch::Tensor pa,torch::Tensor pm,torch::Tensor vs,torch::Tensor out,int hq,int hkv,int length,int splits,int warps,bool pdl);
''',cuda_sources=prefix+body.replace('LANES',str(lanes))+host,functions=['reduce'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-Xptxas=-v'],extra_cflags=['-O3'],verbose=False)
    return MODS[lanes]
class Wrapper:
    def __init__(self,warps,lanes): self.warps=warps;self.mod=get(lanes)
    def __getitem__(self,grid):
        def run(qkv,pa,pm,vs,out,**kw):
            self.mod.reduce(qkv,pa,pm,vs,out,kw['HQ'],kw['HKV'],kw['L'],kw['S'],self.warps,kw['PDL'])
        return run
