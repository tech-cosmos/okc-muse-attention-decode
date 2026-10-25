# Dump SASS of main's kernels for one scored shape (fresh Triton cache dir), with an opcode census of the hot loop.
import os, sys, tempfile
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="s122tc_")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch, reference, submission
batch = int(sys.argv[1]) if len(sys.argv) > 1 else 32
c = dict(batch=batch, ctx=80000, heads="tp8", layer="full", page=1, q_len=8, dist="coherent", spread="agentic")
data = reference.generate_input(seed=99, **c)
submission.kernel_fn(data); torch.cuda.synchronize()
print(os.environ["TRITON_CACHE_DIR"])
