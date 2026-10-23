Do not use these first NCU profiles as clean timing evidence. The driver held
/tmp/okc-gpu.lock, but a benchmark process had also been started and its warmup
and graph capture happen before taking that timing lock. Overlap cannot be
excluded. All profiles were recollected sequentially after both full benchmark
processes and all correctness/sanitizer processes completed. The final profiles
are in the parent directory. Scored-shape comparisons in compare*.json preceded
all these profiling processes and were not affected.
