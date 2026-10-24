import sys
sys.path.insert(0,'.')
from tools import bench
from experiments.sliding_fp8pairs import candidate
bench.submission.kernel_fn=candidate.kernel_fn
assert bench.test()==0
bench.bench()
