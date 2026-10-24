"""Correctness of a plan override on verify 32 full + the Gluon verify kernels' regs/spills/smem/warps and setmaxnreg lines: regs.py '<json plan>'"""
import json, sys, torch
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, reference, submission
P = json.loads(sys.argv[1]); orig = submission._plan
submission._plan = lambda c, m, d: {**orig(c, m, d), **P}
for ld, c in bench.SCORED:
    if f"{ld} {c['layer']}" != "verify 32 full": continue
    data = reference.generate_input(seed=1, **c)
    out = submission.kernel_fn(data); torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    print("PASS" if ok else "FAIL", msg[:80])
for kf in (submission._gw_verify, submission._gw_verify3n):
 for dev, cache in kf.device_caches.items():
    for k, kern in cache[0].items():
        print(kf.fn.__name__ if hasattr(kf,"fn") else "", kern.n_regs, kern.n_spills, kern.metadata.shared, kern.metadata.num_warps)
import re
for dev, cache in submission._gw_verify3n.device_caches.items():
    for k, kern in cache[0].items():
        print([l.strip() for l in kern.asm["ptx"].splitlines() if "setmaxnreg" in l][:12])
import subprocess, tempfile
for dev, cache in submission._gw_verify3n.device_caches.items():
    for k, kern in cache[0].items():
        f = tempfile.NamedTemporaryFile(suffix=".cubin", delete=False); f.write(kern.asm["cubin"]); f.close()
        sass = subprocess.run(["/usr/local/cuda-13.0/bin/cuobjdump", "-sass", f.name], capture_output=True, text=True).stdout
        open("/tmp/s51_v3n.sass", "w").write(sass)
        print("STL", sass.count("STL"), "LDL", sass.count("LDL"))
