from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'previous.py').read_text()
old=''' x=x/den*vscale;y=y/den*vscale;
 float sq=sum32(x*x+y*y),r=rsqrtf(sq/64.f+1e-5f);'''
new=''' // RMSNorm(a * vscale / den), with the epsilon retained exactly in
 // real arithmetic. The vector sum of squares no longer depends on den.
 float sq=sum32(x*x+y*y);
 float r=vscale*rsqrtf(fmaf(sq/64.f,vscale*vscale,1e-5f*den*den));'''
assert s.count(old)==1
s=s.replace(old,new).replace('name="muse_native_merge"','name="muse_native_merge_homogeneous_s38"')
(root/'candidate.py').write_text(s)
