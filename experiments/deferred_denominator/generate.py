"""Remove only the per-tile cross-warp denominator reduction.

All warps use the same running maximum for each query, hence the same alpha.
Their local denominator recurrences can be summed once after the final tile.
The attention numerator, maxima, precision, gathers and grid remain unchanged.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def generate(lane_local=False):
    source = (ROOT / 'natural_v.cu').read_text()
    start = source.index('  // All threads finish reading maxima')
    end = source.index('  asm volatile("fence.proxy.async.shared::cta;', start)
    source = source[:start] + '''  // Each warp retains its own denominator contribution. The maximum and
  // alpha are shared across warps, so cross-warp summation can wait until exit.
  #pragma unroll
  for(int h=0;h<2;h++){
   #pragma unroll
   for(int c=0;c<2;c++){
    float s=score[h][c]+score[h][c+2];
    #pragma unroll
    for(int d=4;d<=16;d*=2)s+=__shfl_xor_sync(0xffffffff,s,d);
    sum[h][c]+=s;
   }
  }
''' + source[end:]
    start = source.index('  #pragma unroll\n  for(int h=0;h<2;h++){', source.index('wgmma.wait_group.sync.aligned 0;":::"memory");pin(acc[0])'))
    end = source.index(' #pragma unroll\n for(int h=0;h<2;h++){', start)
    source = source[:start] + ''' }
 asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
 #pragma unroll
 for(int h=0;h<2;h++){
  #pragma unroll
  for(int c=0;c<2;c++)if(lane<4)stats[h][warp][lane*2+c]=sum[h][c];
 }
 __syncthreads();
 #pragma unroll
 for(int h=0;h<2;h++){
  #pragma unroll
  for(int c=0;c<2;c++){
   sum[h][c]=0.f;
   #pragma unroll
   for(int w=0;w<4;w++)sum[h][c]+=stats[h][w][lane%4*2+c];
  }
 }
''' + source[end:]
    if lane_local:
        loop = '''    #pragma unroll
    for(int d=4;d<=16;d*=2)s+=__shfl_xor_sync(0xffffffff,s,d);
    sum[h][c]+=s;'''
        assert source.count(loop) == 1
        source = source.replace(loop, '    sum[h][c]+=s;')
        store = '  for(int c=0;c<2;c++)if(lane<4)stats[h][warp][lane*2+c]=sum[h][c];'
        assert source.count(store) == 1
        source = source.replace(store, '''  for(int c=0;c<2;c++){
   float s=sum[h][c];
   #pragma unroll
   for(int d=4;d<=16;d*=2)s+=__shfl_xor_sync(0xffffffff,s,d);
   if(lane<4)stats[h][warp][lane*2+c]=s;
  }''')
    return source

if __name__ == '__main__':
    (ROOT / 'deferred.cu').write_text(generate())
    (ROOT / 'lane.cu').write_text(generate(lane_local=True))
