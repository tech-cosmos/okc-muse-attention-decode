import sys
sys.path.insert(0,'.')
from tools import bench
from experiments.sliding_stores import candidate
bench.submission.kernel_fn=candidate.kernel_fn
bench.bench()
