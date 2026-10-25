import sys; sys.path.insert(0, "/home/daytona/repo")
import reference as R, torch, importlib.util as U
sp = U.spec_from_file_location("s", sys.argv[1]); submission = U.module_from_spec(sp); sp.loader.exec_module(submission)
layer, ql = sys.argv[2], int(sys.argv[3])
c = dict(batch=1, ctx=int(sys.argv[4]) if len(sys.argv) > 4 else 80000, heads="tp8", layer=layer, page=1, q_len=ql, dist="coherent", spread="agentic")
d = R.generate_input(seed=99, **c)
out = submission.kernel_fn(d); torch.cuda.synchronize()
print(layer, ql, d["seq_lens"].tolist(), d["page_table"].shape[1], R.check_implementation(d, R.collect(d, out))[1][:50], flush=True)
