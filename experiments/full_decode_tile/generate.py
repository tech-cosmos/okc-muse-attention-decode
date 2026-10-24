"""Build full L1 single-tile native kernels from frozen production CUDA."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parent
source=(ROOT/'production_snapshot.py').read_text()
tree=ast.parse(source)
call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_SLIDING' for t in n.targets))
base=next(ast.literal_eval(k.value) for k in call.keywords if k.arg=='cuda_sources')
cpp=next(ast.literal_eval(k.value) for k in call.keywords if k.arg=='cpp_sources')
for bn in (128,256):
    s=base
    s=s.replace('unsigned char K[4096]',f'unsigned char K[{bn*64}]').replace('unsigned char Vraw[4096]',f'unsigned char Vraw[{bn*64}]').replace('half V[4096]',f'half V[{bn*64}]')
    s=s.replace('base=max(0,seq-2047),lo=base+split*64','base=0,lo=split*'+str(bn))
    s=s.replace('j<2;j++',f'j<{bn//32};j++')
    a=s.index(' float c0=');z=s.index('\n unsigned qr',a);s=s[:a]+s[z:]
    a=s.index('     float cc=');z=s.index('     packed|=',a)
    s=s[:a]+'     float a=x[ix]*inv,bb=x[ix+1]*inv;\n'+s[z:]
    s=s.replace('base=0,lo=', 'lo=')
    s=s.replace('int col=d*32+(lane%4)*4+i*16+k,ix=', 'int ix=')
    s=s.replace('seq<lo+64',f'seq<lo+{bn}')
    a=s.index('  float xp=');z=s.index('  nv0=',a);s=s[:a]+'  nx*=inv;ny*=inv;\n'+s[z:]
    s=s.replace('for(int j=0;j<4;j++) {\n  unsigned i=8*tid',f'for(int j=0;j<{bn//16};j++) {{\n  unsigned i=8*tid')
    # Replace the QK accumulator and its asynchronous register fences/asm.
    s=s.replace('float score[32]',f'float score[{bn//2}]')
    def fence(var,n):
        return ' asm volatile("" : '+','.join(f'"+f"({var}[{i}])' for i in range(n))+' :: "memory");'
    lines=s.splitlines()
    for i,line in enumerate(lines):
        if 'asm volatile("" : "+f"(score[0])' in line:lines[i]=fence('score',bn//2)
        if 'asm volatile("wgmma.mma_async.sync.aligned.m64n64k32' in line:
            n=bn//2
            regs='{'+','.join('%'+str(i) for i in range(n))+'}'
            qr='{'+','.join('%'+str(n+i) for i in range(4))+'}'
            constraint=','.join(f'"+f"(score[{i}])' for i in range(n))
            inputs=','.join(f'"r"(qr[d][{i}])' for i in range(4))+',"l"(desc)'
            lines[i]=f'  asm volatile("wgmma.mma_async.sync.aligned.m64n{bn}k32.f32.e4m3.e4m3 {regs}, {qr}, %{n+4}, 1, 1, 1;" : {constraint} : {inputs});'
    s='\n'.join(lines)+'\n'
    # Only QK/softmax have BN/2 elements; PV still produces 64 dimensions.
    a=s.index(' float mx[2]');z=s.index(' float acc[32]',a)
    part=s[a:z].replace('i<32;i++',f'i<{bn//2};i++').replace('pr[4][4]',f'pr[{bn//16}][4]').replace('d<4;d++',f'd<{bn//16};d++')
    s=s[:a]+part+s[z:]
    # PV reduction covers every one of the BN positions.
    a=s.index(' float acc[32]');z=s.index(' if(warp==0)',a)
    s=s[:a]+s[a:z].replace('d<4;d++',f'd<{bn//16};d++')+s[z:]
    # Keep every register-source probability alive until asynchronous PV completes.
    a=s.index(' float acc[32]')
    z=s.index(' if(warp==0)',a)
    s=s[:z]+' asm volatile("" :: '+','.join(f'"r"(pr[{d}][{i}])' for d in range(bn//16) for i in range(4))+' : "memory");\n'+s[z:]
    a=s.index(' __shared__ __align__(512) unsigned char K[');z=s.index(' const int tid=',a)
    s=s[:a]+f' extern __shared__ __align__(1024) unsigned char storage[];\n unsigned char* K=storage;unsigned char* Vraw=storage+{bn*64};half* V=(half*)(storage+{bn*128});\n'+s[z:]
    s=s.replace('sliding<P,E><<<grid,128,0,stream>>>',f'sliding<P,E><<<grid,128,{bn*256},stream>>>')
    s=s.replace(' #define RUN(P,E)',f' #define RUN(P,E) C10_CUDA_CHECK(cudaFuncSetAttribute(sliding<P,E>,cudaFuncAttributeMaxDynamicSharedMemorySize,{bn*256}));')
    # Full attention does not use rope; keep binding shape for direct comparison.
    (ROOT/f'tile{bn}.cu').write_text(s)
    (ROOT/'binding.cpp').write_text(cpp)
