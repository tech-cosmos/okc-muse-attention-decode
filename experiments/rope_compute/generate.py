"""Replace only native RoPE lookup, keeping production math and launch topology."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
source=(ROOT/'production_snapshot.py').read_text()
tree=ast.parse(source)
call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_SLIDING' for t in n.targets))
args={k.arg:ast.literal_eval(k.value) for k in call.keywords}
cuda=args['cuda_sources']
helper='''// Match the table's FP32 angle multiplication, then reduce that angle in FP64.
// Hardware sin/cos receive only [-pi,pi]; no large-argument fast-intrinsic error.
__device__ __forceinline__ float2 rope_pair(int position,float frequency) {
 float angle=__fmul_rn((float)position,frequency);
 double turns=(double)__double2int_rn((double)angle*0.15915494309189533577);
 float reduced=__double2float_rn(__fma_rn(-turns,6.28318530717958647693,(double)angle));
 float2 result;
 asm("cos.approx.ftz.f32 %0,%1;":"=f"(result.x):"f"(reduced));
 asm("sin.approx.ftz.f32 %0,%1;":"=f"(result.y):"f"(reduced));
 return result;
}
'''
cuda=cuda.replace('// EARLY only changes',helper+'// EARLY only changes',1)
old=''' float c0=cs[(long)seq*64+lane/2],s0=cs[(long)seq*64+lane/2+32];
 float c1=cs[(long)seq*64+(lane+32)/2],s1=cs[(long)seq*64+(lane+32)/2+32];'''
new=''' float2 r0=rope_pair(seq,cs[lane/2]),r1=rope_pair(seq,cs[(lane+32)/2]);
 float c0=r0.x,s0=r0.y,c1=r1.x,s1=r1.y;'''
assert cuda.count(old)==1;cuda=cuda.replace(old,new)
old='float cc=cs[(long)seq*64+col/2],ss=cs[(long)seq*64+col/2+32];'
new='float2 rot=rope_pair(seq,cs[col/2]);float cc=rot.x,ss=rot.y;'
assert cuda.count(old)==1;cuda=cuda.replace(old,new)
probe='''
__global__ void probe_kernel(const float* freq,float* out,int start,int count) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<count*32) {
  float2 r=rope_pair(start+i/32,freq[i%32]);
  out[2*i]=r.x;out[2*i+1]=r.y;
 }
}
void rope_probe(at::Tensor freq,at::Tensor out,int start,int count) {
 auto stream=c10::cuda::getCurrentCUDAStream();
 probe_kernel<<<(count*32+255)/256,256,0,stream>>>(freq.data_ptr<float>(),out.data_ptr<float>(),start,count);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
cuda+=probe
binding=args['cpp_sources']+'\nvoid rope_probe(torch::Tensor freq,torch::Tensor out,int start,int count);\n'
candidate=source.replace(args['cuda_sources'],cuda,1).replace(args['cpp_sources'],binding,1)
candidate=candidate.replace('muse_native_sliding_direct_v1','muse_native_rope_compute_s34')
candidate=candidate.replace('functions=["attention"]','functions=["attention", "rope_probe"]',1)
freq='''_FREQUENCIES = {}
def _rope_freq(theta,device):
    key=(theta,device)
    if key not in _FREQUENCIES:
        _FREQUENCIES[key]=(1.0/(theta**(torch.arange(0,64,2,dtype=torch.float32,device=device)/64))).contiguous()
    return _FREQUENCIES[key]

'''
candidate=candidate.replace('_ROPE = {}',freq+'_ROPE = {}',1)
old='cs = _cos_sin(cfg["rope_theta"], 64, device)'
assert candidate.count(old)==1
candidate=candidate.replace(old,'cs = _rope_freq(cfg["rope_theta"], device)',1)
(ROOT/'candidate.cu').write_text(cuda)
(ROOT/'parent.cu').write_text(args['cuda_sources'])
(ROOT/'binding.cpp').write_text(binding)
(ROOT/'candidate.py').write_text(candidate)
(ROOT/'helper.cuh').write_text(helper)
old=''' double turns=(double)__double2int_rn((double)angle*0.15915494309189533577);
 float reduced=__double2float_rn(__fma_rn(-turns,6.28318530717958647693,(double)angle));'''
new=''' // Split 2*pi into a rounded float and its rounded residual. FMA retains
 // cancellation accuracy without float64 arithmetic/conversions.
 float turns=__int2float_rn(__float2int_rn(__fmul_rn(angle,0.15915494309189533577f)));
 float reduced=__fmaf_rn(-turns,6.283185482025146484375f,angle);
 reduced=__fmaf_rn(-turns,-1.7484555314695172e-7f,reduced);'''
assert cuda.count(old)==1
compensated=cuda.replace(old,new)
(ROOT/'compensated.cu').write_text(compensated)
(ROOT/'compensated.py').write_text(candidate.replace(cuda,compensated,1).replace('muse_native_rope_compute_s34','muse_native_rope_compensated_s34'))
(ROOT/'results'/'source-manifest.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('production_snapshot.py','candidate.py','candidate.cu','compensated.py','compensated.cu','parent.cu','helper.cuh')},indent=2)+'\n')
