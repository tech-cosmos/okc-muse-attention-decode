import sys, re, subprocess, os, tempfile
src_file, half = sys.argv[1], sys.argv[2]
s = open(src_file).read(); a = s.index("_NVS_SRC = r'''") + len("_NVS_SRC = r'''"); b = s.index("'''", a)
src = s[a:b]
if half != "-": src = re.sub(r"#define NV_HALF \d", f"#define NV_HALF {half}", src)
d = tempfile.mkdtemp(); open(f"{d}/k.cu", "w").write(src)
import torch.utils.cpp_extension as ce
inc = " ".join("-I" + p for p in ce.include_paths(device_type="cuda"))
import sysconfig
cmd = f"/usr/local/cuda/bin/nvcc -c {d}/k.cu -o {d}/k.o -gencode=arch=compute_90a,code=sm_90a -O3 -std=c++20 -Xptxas -v -I{sysconfig.get_paths()['include']} {inc} -D_GLIBCXX_USE_CXX11_ABI=1"
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(r.returncode, r.stderr[-3000:])
