"""Raw-cache layout experiments, based on session10's synchronous PV SW128."""
from pathlib import Path
p=Path(__file__).resolve().parent
base=(p/'baseline.py').read_text()
def save(name,s):
 s=s.replace("'.build-baseline'",repr('.build-'+name)).replace("name='codex_key_baseline'",'name='+repr('codex_key_'+name))
 (p/(name+'.py')).write_text(s)
rawv=base.replace('row_in_tile*(D+16)+piece*16','row_in_tile*D+piece*16').replace('(ii/D)*(D+16)+ii%D','ii').replace('(i/D)*(D+16)+i%D','i')
assert rawv!=base
save('rawv',rawv)
# SW64 K-major: contiguous 64-byte row, XOR byte bits [8:7] into [5:4].
# K/QK descriptor uses 512-byte stride across 8 rows; 2nd K32 starts at +32.
for name,src in [('key64',base),('rawvk64',rawv)]:
 s=src.replace('__align__(16) unsigned char raw','__align__(512) unsigned char raw')
 marker='template<int STAGES, int PAGE>'
 s=s.replace(marker,'__device__ __forceinline__ unsigned kindex(unsigned i) { return i ^ (((i>>7)&3)<<4); }\n'+marker)
 s=s.replace('(row_in_tile/8)*512+piece*128+(row_in_tile%8)*16','kindex(row_in_tile*D+piece*16)')
 s=s.replace('(ii/(D*8))*512+((ii%D)/16)*128+((ii/D)%8)*16+ii%16','kindex(ii)')
 s=s.replace('raw[stage][wg][0]+d*256','raw[stage][wg][0]+d*32')
 s=s.replace('| (8ull<<16) | (32ull<<32);','| (1ull<<16) | (32ull<<32) | (2ull<<62);')
 save(name,s)
# Store two K heads in one SW128 tile. Both heads' gather destinations now share
# a row; each WGMMA descriptor selects its own 64 K columns without extra bytes.
s=(p/'rawvk64.py').read_text()
s=s.replace("'.build-rawvk64'","'.build-paired128'").replace("name='codex_key_rawvk64'","name='codex_key_paired128'")
s=s.replace('return i ^ (((i>>7)&3)<<4);','return i ^ (((i>>7)&7)<<4);')
s=s.replace('__shared__ __align__(512) unsigned char raw[STAGES][2][2][N*(D+16)];','__shared__ __align__(1024) unsigned char rawK[STAGES][N*2*D];\n __shared__ __align__(16) unsigned char rawV[STAGES][2][N*D];')
s=s.replace('raw[stage][hh][0]+kindex(row_in_tile*D+piece*16)','rawK[stage]+kindex(row_in_tile*2*D+hh*D+piece*16)')
s=s.replace('raw[stage][hh][0][kindex(ii)]','rawK[stage][kindex((ii/D)*2*D+hh*D+ii%D)]')
s=s.replace('raw[stage][wg][0]+d*32','rawK[stage]+wg*D+d*32')
s=s.replace('raw[stage][hh][1]','rawV[stage][hh]').replace('raw[stage][wg][1]','rawV[stage][wg]')
s=s.replace('| (1ull<<16) | (32ull<<32) | (2ull<<62);','| (1ull<<16) | (64ull<<32) | (1ull<<62);')
assert 'raw[' not in s
(p/'paired128.py').write_text(s)
