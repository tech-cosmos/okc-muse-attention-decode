from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
s=(ROOT/'production_snapshot.py').read_text()
a='const int b=blockIdx.y,h=blockIdx.z,split=blockIdx.x;'
b='dim3 grid(S,qkv.size(0),hkv);'
assert s.count(a)==1 and s.count(b)==1
v=s.replace(a,'const int b=blockIdx.z,h=blockIdx.x,split=blockIdx.y;').replace(b,'dim3 grid(hkv,S,qkv.size(0));').replace('muse_native_sliding_direct_v1','muse_native_sliding_head_neighbor_s30')
(ROOT/'neighbor.py').write_text(v)
(ROOT/'results/source-manifest.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['production_snapshot.py','neighbor.py']},indent=2)+'\n')
