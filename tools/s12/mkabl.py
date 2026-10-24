s=open('/home/daytona/repo/submission.py').read()
i=s.index("def _gw_producer_cv"); j=s.index("def _gw_consumer_cv")
seg=s[i:j]
seg2=seg.replace("                cp.async_copy_global_to_shared(k_ring.index(s), kbase + off)\n","").replace("                cp.async_copy_global_to_shared(v_ring.index(s), vbase + off)\n","")
assert seg2!=seg
open('/tmp/s12/abl/submission.py','w').write(s[:i]+seg2+s[j:])
