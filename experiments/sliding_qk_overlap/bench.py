import sys
sys.path.insert(0,'.')
from tools import bench
from experiments.sliding_qk_overlap import candidate
bench.submission.kernel_fn=candidate.kernel_fn
bench.bench()
