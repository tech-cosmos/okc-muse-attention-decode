"""Install the selected S65..128, 8-warp full-layer merge into production source."""
from pathlib import Path
import ast
s=Path('submission.py').read_text()
mod=ast.parse(Path('experiments/reduce/native.py').read_text())
cuda=next(ast.literal_eval(n.value) for n in mod.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA' for t in n.targets))
a=cuda.index('__device__ __forceinline__ float2 read2');b=cuda.index('void reduce(at::Tensor')
kernel=cuda[a:b]
host=r'''
void combine_full(at::Tensor qkv,at::Tensor pa,at::Tensor pm,at::Tensor vs,at::Tensor out,int hq,int hkv,int length,int splits,bool pdl) {
 auto stream=c10::cuda::getCurrentCUDAStream();int b=qkv.size(0);
 cudaLaunchConfig_t config={};config.gridDim=dim3(b*length*hq);config.blockDim=dim3(256);config.stream=stream;
 cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=pdl;
 config.attrs=&attr;config.numAttrs=1;
 #define FULL_RUN(P) C10_CUDA_CHECK(cudaLaunchKernelEx(&config,reduce_kernel<8,16,P>,(const __nv_bfloat16*)qkv.data_ptr(),pa.data_ptr<float>(),(const float2*)pm.data_ptr<float>(),vs.data_ptr<float>(),(__nv_bfloat16*)out.data_ptr(),hq,hkv,length,b,splits))
 if(pdl){FULL_RUN(true);}else{FULL_RUN(false);}
 #undef FULL_RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
decl='void combine(torch::Tensor qkv,torch::Tensor pa,torch::Tensor pm,torch::Tensor vs,torch::Tensor out,int hq,int hkv,int length,int splits,bool pdl);'
s=s.replace(decl,decl+'\n'+decl.replace('void combine(', 'void combine_full('))
s=s.replace('template<int S,int W,bool PDL>',kernel+host+'\ntemplate<int S,int W,bool PDL>',1)
s=s.replace('functions=["combine"],','functions=["combine", "combine_full"],')
s=s.replace('    hb = plan.get("hb", 1)', '''    # Eight warps independently merge up to 16 FP32 splits, then combine their
    # partial results in shared memory. Larger batches retain the Triton merge.
    if not cfg["window"] and batch * hkv <= 4 and 64 < splits <= 128:
        _NATIVE_MERGE.combine_full(qkv, pacc, pml, data["v_scale"], out,
                                   hq, hkv, length, splits, USE_PDL)
        return out
    hb = plan.get("hb", 1)''')
s=s.replace('native CUDA merge for small sliding reductions.','native CUDA merges for small batches.')
s=s.replace('# A native warp merges one head\'s FP32 partials, then normalizes and gates the output.','# Native reductions merge each head\'s FP32 partials, then normalize and gate its output.')
Path('submission.py').write_text(s)
