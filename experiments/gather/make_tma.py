from pathlib import Path
s=Path('experiments/gather/pair_mbar.py').read_text()
s=s.replace('__launch_bounds__(320,2)','__launch_bounds__(288,2)').replace('<<<grid,320,0,stream>>>','<<<grid,288,0,stream>>>')
s=s.replace('int HQ,int HKV,int MP,int SPLITS,float SCALE) {','int HQ,int HKV,int MP,int SPLITS,float SCALE,const __grid_constant__ CUtensorMap kmap) {')
s=s.replace('__shared__ __align__(16) unsigned char raw[STAGES][2][2][N*(D+16)];','__shared__ __align__(1024) unsigned char raw[STAGES][2][N*128];')
s=s.replace('mbarrier.init.shared.b64 [%0],64;','mbarrier.init.shared.b64 [%0],1;')
a=s.index('       if(!isnew) {');b=s.index('\n   }\n   return;',a)
s=s[:a]+'''       if(!isnew) {
         if(lane==0)asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"((unsigned)__cvta_generic_to_shared(ready+stage)),"r"(N*128*2):"memory");
         __syncwarp();
         int t=min(lo+step*N+lane,hi-1);
         int row=pt[b*MP+t/PAGE]*PAGE+t%PAGE;
         unsigned smk=__cvta_generic_to_shared(raw[stage][0]+lane*128),smv=__cvta_generic_to_shared(raw[stage][1]+lane*128),bar=__cvta_generic_to_shared(ready+stage);
         asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"::"r"(smk),"l"(&kmap),"r"(blockIdx.z*128),"r"(row),"r"(bar):"memory");
         asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],128,[%2];"::"r"(smv),"l"(vc+(long)row*HKV*D+blockIdx.z*128),"r"(bar):"memory");
       } else {
         for(int i=lane;i<N*128;i+=32) {
           unsigned rr=i/128,hh=(i%128)/64,dd=i%64;
           raw[stage][0][i^((rr&7)<<4)]=rr<8?own[hh][0][rr*64+dd]:0;
           raw[stage][1][i]=rr<8?own[hh][1][rr*64+dd]:0;
         }
         __syncwarp();
         asm volatile("fence.proxy.async.shared::cta;":::"memory");
         if(lane==0)mbarrive(ready+stage);
       }'''+s[b:]
s=s.replace('raw[stage][wg][1]+(i/D)*(D+16)+i%D','raw[stage][1]+(i/D)*128+wg*64+i%D')
s=s.replace('raw[stage][wg][0]+d*256','raw[stage][0]+wg*64+d*32')
s=s.replace('| (8ull<<16) | (32ull<<32)','| (1ull<<16) | (64ull<<32) | (1ull<<62)')
a=s.index(' auto stream=c10::cuda::getCurrentCUDAStream();')
s=s[:a]+''' CUtensorMap kmap;
 cuuint64_t dims[2]={(cuuint64_t)hkv*64,(cuuint64_t)kc.size(0)*kc.size(1)};
 cuuint64_t strides[1]={(cuuint64_t)hkv*64};
 cuuint32_t box[2]={128,1},element[2]={1,1};
 CUresult result=cuTensorMapEncodeTiled(&kmap,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,kc.data_ptr(),dims,strides,box,element,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 TORCH_CHECK(result==CUDA_SUCCESS,"cuTensorMapEncodeTiled failed: ",int(result));
'''+s[a:]
s=s.replace('pt.size(1),splits,scale)','pt.size(1),splits,scale,kmap)')
s=s.replace("'.build-pair-mbar'","'.build-pair-tma'").replace('codex_verify_pair3mbar','codex_verify_pair3tma')
s=s.replace("extra_cflags=['-O3'], verbose=True)","extra_cflags=['-O3'], extra_ldflags=['-lcuda'], verbose=True)")
Path('experiments/gather/pair_tma.py').write_text(s)
s=Path('experiments/gather/bench_pair_mbar.py').read_text().replace('pair_mbar','pair_tma').replace('pair-mbar-','pair-tma-').replace('(3,4,5),(132,264,528)','(2,3,4),(132,264,528,1056)')
Path('experiments/gather/bench_pair_tma.py').write_text(s)
