"""Prepare Q/new KV once per head, overlapping its prologue with every tile's gather."""
from pathlib import Path
R=Path(__file__).resolve().parent
s=(R/'vector.cu').read_text()
q=s[s.index(' unsigned qr'):s.index(' // Each warp prepares')]
k=s[s.index(' // Each warp prepares'):s.index(' if(!EARLY)gather();')]
k=k.replace('own[j]=pos>=lo && pos<lo+64;', 'own[j]=true;')
write=s[s.index(' #pragma unroll\n for(int j=0;j<2;j++)if(own[j])'):s.index(' __syncthreads();\n #pragma unroll\n for(int j=0;j<4;j++)')]
write=write.replace('int token=warp+4*j,row=seq+token-lo;', 'int token=warp+4*j;')
write=write.replace('  K[sw64(row*64+lane)]=k0;K[sw64(row*64+lane+32)]=k1;\n  Vraw[row*64+lane]=v0;Vraw[row*64+lane+32]=v1;', '''  long ni=((long)(b*HKV+h)*8+token)*128;
  nkv[ni+lane]=k0;nkv[ni+lane+32]=k1;nkv[ni+64+lane]=v0;nkv[ni+96+lane]=v1;''')
prep='''__global__ __launch_bounds__(128) void prepare(const __nv_bfloat16* qkv,const float* cs,
 unsigned char* kc,unsigned char* vc,const int* seqs,const long* loc,
 const float* ks,const float* vs,unsigned* pq,unsigned char* nkv,int HQ,int HKV) {
 const int tid=threadIdx.x,warp=tid/32,lane=tid%32,b=blockIdx.x,h=blockIdx.y;
 const int seq=seqs[b],width=(2*HQ+2*HKV)*64;
 asm volatile("griddepcontrol.launch_dependents;":::"memory");
 float kscale=*ks,vscale=*vs;
''' + q + '''
 #pragma unroll
 for(int d=0;d<2;d++) {
  unsigned* p=pq+((long)(b*HKV+h)*128+tid)*8+d*4;
  *(uint4*)p=make_uint4(qr[d][0],qr[d][1],qr[d][2],qr[d][3]);
 }
''' + k + write+'}\n'
s=s.replace('// EARLY only changes when cached-key copies are issued, never the arithmetic.',prep+'\n// EARLY changes whether the historical gather precedes the predecessor wait.')
s=s.replace('const float* ks,const float* vs,float* pa,float* pm,int HQ', 'const float* ks,const float* vs,const unsigned* pq,const unsigned char* nkv,float* pa,float* pm,int HQ')
a=s.index(' float kscale=*ks,vscale=*vs;',s.index('void sliding'));b=s.index(' __syncthreads();\n #pragma unroll\n for(int j=0;j<4;j++)',a)
s=s[:a]+''' // Q/new KV are produced by the previous kernel; cached gathers are independent.
 asm volatile("griddepcontrol.wait;":::"memory");
 float kscale=*ks;
 unsigned qr[2][4];
 #pragma unroll
 for(int d=0;d<2;d++) {
  const unsigned* p=pq+((long)(b*HKV+h)*128+tid)*8+d*4;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(qr[d][0]),"=r"(qr[d][1]),"=r"(qr[d][2]),"=r"(qr[d][3]):"l"(p):"memory");
 }
 if(!EARLY)gather();
 // Avoid overlapping zero-fill and own-token shared writes.
 asm volatile("cp.async.wait_group 0;":::"memory");
 __syncthreads();
 if(tid<32) {
  int token=tid/4,piece=(tid%4)*16,row=seq+token-lo;
  if(row>=0 && row<64) {
   const unsigned char* src=nkv+((long)(b*HKV+h)*8+token)*128+piece;
   cp16(K+sw64(row*64+piece),src,1);
   cp16(Vraw+row*64+piece,src+64,1);
  }
 }
 asm volatile("cp.async.commit_group;\\ncp.async.wait_group 0;":::"memory");
''' +s[b:]
s=s.replace('at::Tensor loc,at::Tensor ks,at::Tensor vs,at::Tensor pa,at::Tensor pm,int page,double scale,bool early)', 'at::Tensor loc,at::Tensor ks,at::Tensor vs,at::Tensor pq,at::Tensor nkv,at::Tensor pa,at::Tensor pm,int page,double scale,bool early)')
s=s.replace(' #define RUN(P,E) sliding<P,E><<<grid,128,0,stream>>>(', ''' prepare<<<dim3(qkv.size(0),hkv),128,0,stream>>>((const __nv_bfloat16*)qkv.data_ptr(),cs.data_ptr<float>(),(unsigned char*)kc.data_ptr(),(unsigned char*)vc.data_ptr(),seq.data_ptr<int>(),loc.data_ptr<long>(),ks.data_ptr<float>(),vs.data_ptr<float>(),(unsigned*)pq.data_ptr(),(unsigned char*)nkv.data_ptr(),hq,hkv);
 cudaLaunchConfig_t config={};config.gridDim=grid;config.blockDim=dim3(128);config.stream=stream;
 cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;
 config.attrs=&attr;config.numAttrs=1;
 #define RUN(P,E) C10_CUDA_CHECK(cudaLaunchKernelEx(&config,sliding<P,E>,''')
s=s.replace('vs.data_ptr<float>(),pa.data_ptr<float>()', 'vs.data_ptr<float>(),(const unsigned*)pq.data_ptr(),(const unsigned char*)nkv.data_ptr(),pa.data_ptr<float>()')
s=s.replace('hq,hkv,pt.size(1),S,scale)\n if(page', 'hq,hkv,pt.size(1),S,(float)scale))\n if(page')
(R/'precomputed.cu').write_text(s)
s=(R/'native.py').read_text().replace("/'.build'","/'.build-precomputed'").replace('codex_sliding_verify','codex_sliding_verify_precomputed').replace("ROOT/'kernel.cu'","ROOT/'precomputed.cu'")
s=s.replace('torch::Tensor vs,torch::Tensor pa,torch::Tensor pm', 'torch::Tensor vs,torch::Tensor pq,torch::Tensor nkv,torch::Tensor pa,torch::Tensor pm')
s=s.replace(' module.attention('," pq=torch.empty((b,h,128,8),dtype=torch.int32,device=device)\n nkv=torch.empty((b,h,8,128),dtype=torch.uint8,device=device)\n module.attention(")
s=s.replace("data['v_scale'],pa,pm", "data['v_scale'],pq,nkv,pa,pm")
(R/'precomputed.py').write_text(s)
