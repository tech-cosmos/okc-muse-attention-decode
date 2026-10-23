"""Reproduce the paired-head prototype from the archived session-1 implementation."""
from pathlib import Path
s=Path('experiments/gather/single_head.py').read_text()
s=s.replace('Hopper native attention experiment; slower than submission.py, never dispatched by it.','Hopper paired-head native attention experiment; never dispatched by submission.py.')
s=s.replace('__device__ __forceinline__ void consumer_barrier() {\n asm volatile("bar.sync 1, 128;":::"memory");','__device__ __forceinline__ void consumer_barrier(unsigned wg) {\n asm volatile("bar.sync %0, 128;"::"r"(wg+1):"memory");')
s=s.replace('ld.volatile.shared.u32','ld.acquire.cta.shared.u32')
s=s.replace('__launch_bounds__(160,4)','__launch_bounds__(288,2)')
s=s.replace('unsigned char raw[STAGES][2][N*(D+16)];','unsigned char raw[STAGES][2][2][N*(D+16)];')
s=s.replace('unsigned char own[2][8*D];','unsigned char own[2][2][8*D];')
s=s.replace('half V[N*(D+8)];','half V[2][N*(D+8)];')
s=s.replace('unsigned char Q[M*(D+16)];','unsigned char Q[2][M*(D+16)];')
s=s.replace('unsigned ready[STAGES];','unsigned ready[STAGES][2];')
s=s.replace('const unsigned tid=threadIdx.x, warp=tid/32, lane=tid%32;','const unsigned global_tid=threadIdx.x, wg=global_tid/128, tid=global_tid%128, warp=tid/32, lane=tid%32;')
s=s.replace('h=blockIdx.z, split=blockIdx.x','h=blockIdx.z*2+wg, split=blockIdx.x')
s=s.replace('for(int i=tid;i<STAGES;i+=blockDim.x) ready[i]=0;','for(int i=global_tid;i<STAGES*2;i+=blockDim.x) ((unsigned*)ready)[i]=0;')
s=s.replace('for(int row=warp;row<M;row+=5)', 'if(wg<2) for(int row=warp;row<M;row+=4)')
s=s.replace('if(split==0) for(int row=warp;row<8;row+=5)', 'if(wg<2 && split==0) for(int row=warp;row<8;row+=4)')
s=s.replace('Q[row*','Q[wg][row*')
s=s.replace('own[0][row','own[wg][0][row').replace('own[1][row','own[wg][1][row')
s=s.replace('if(warp==4)', 'if(wg==2)')
s=s.replace('if(lane==0) release(ready+consumed%STAGES,1);','if(lane==0) { release(&ready[consumed%STAGES][0],1); release(&ready[consumed%STAGES][1],1); }')
s=s.replace('while(acquire(ready+stage)!=0) {}','while(acquire(&ready[stage][0])!=0 || acquire(&ready[stage][1])!=0) {}')
start=s.index('         for(int i=lane;i<N*4;i+=32)')
end=s.index('\n       asm volatile("cp.async.commit_group;',start)
s=s[:start]+'''         for(int i=lane;i<N*8;i+=32) {
           unsigned row_in_tile=i/8, hh=(i%8)/4, piece=i%4;
           int t=min(lo+step*N+row_in_tile,hi-1);
           long row=((long)pt[b*MP+t/PAGE]*PAGE+t%PAGE)*HKV*D+blockIdx.z*2*D;
           copy16(raw[stage][hh][0]+(row_in_tile/8)*512+piece*128+(row_in_tile%8)*16,kc+row+(i%8)*16);
           copy16(raw[stage][hh][1]+row_in_tile*(D+16)+piece*16,vc+row+(i%8)*16);
         }
       } else {
         for(int i=lane;i<2*N*D;i+=32) {
           unsigned hh=i/(N*D), ii=i%(N*D);
           raw[stage][hh][0][(ii/(D*8))*512+((ii%D)/16)*128+((ii/D)%8)*16+ii%16]=ii<8*D?own[hh][0][ii]:0;
           raw[stage][hh][1][(ii/D)*(D+16)+ii%D]=ii<8*D?own[hh][1][ii]:0;
         }
       }'''+s[end:]
s=s.replace('(Q+row*','(Q[wg]+row*')
s=s.replace('while(acquire(ready+stage)!=1)', 'while(acquire(&ready[stage][wg])!=1)')
s=s.replace('raw[stage][1]+','raw[stage][wg][1]+').replace('raw[stage][0]+','raw[stage][wg][0]+')
s=s.replace('(V+target)','(V[wg]+target)').replace('(V+(d*','(V[wg]+(d*')
s=s.replace('consumer_barrier();','consumer_barrier(wg);')
s=s.replace('((unsigned long)__cvta_generic_to_shared(raw[stage][wg][0]+d*256)>>4)','(((unsigned long)__cvta_generic_to_shared(raw[stage][wg][0]+d*256)&0x3ffff)>>4)')
s=s.replace('if(tid==0) release(ready+stage,0);','if(tid==0) release(&ready[stage][wg],0);')
s=s.replace('qkv.size(0),hkv);','qkv.size(0),hkv/2);').replace('<<<grid,160,0,stream>>>','<<<grid,288,0,stream>>>')
# Keep shared footprint under 48 KiB without an opt-in for first experiments.
s=s.replace('LAUNCH(5,','LAUNCH(3,').replace('LAUNCH(6,','LAUNCH(3,').replace('LAUNCH(7,','LAUNCH(3,')
s=s.replace("parent / '.build'", "parent / '.build-pair'")
s=s.replace("name='codex_verify_exp'", "name='codex_verify_pair3'")
s=s.replace('s=splits or max(1,528//(b*h))','s=splits or max(1,264//(b*h//2))')
s=s.replace("assert c['q_len']==8", "assert h%2==0 and c['q_len']==8")
Path('experiments/gather/pair.py').write_text(s)
