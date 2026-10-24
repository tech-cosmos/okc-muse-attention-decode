import sys
sys.path.insert(0,'.')
from tools import bench
from experiments.sliding_boundary import candidate,pruned
original=bench.submission.kernel_fn
# Broad experimental dispatch, for consistency with the native correctness suite.
def run(data):
 return pruned.native(data) if candidate.eligible(data) else original(data)
bench.submission.kernel_fn=run
bench.bench()
