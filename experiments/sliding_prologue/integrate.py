from pathlib import Path
root=Path(__file__).resolve().parent
p=Path('submission.py');s=p.read_text();assert '_NATIVE_SLIDING =' not in s
cpp='''#include <torch/extension.h>
void attention(torch::Tensor qkv,torch::Tensor cs,torch::Tensor kc,torch::Tensor vc,torch::Tensor pt,torch::Tensor seq,torch::Tensor loc,torch::Tensor ks,torch::Tensor vs,torch::Tensor pa,torch::Tensor pm,int page,double scale,bool early);
'''
# Same native source used by reference/fresh/sanitizer validation.
cuda=(root/'direct.cu').read_text()
block='\n\n# Small-batch sliding decode: one 64-key tile per block, first gather before\n# norm/RoPE, FP8 queries in registers, and FP32 split partials.\n_NATIVE_SLIDING = load_inline(\n    name="muse_native_sliding_direct_v1",\n    cpp_sources=r\'\'\''+cpp+"'''"+',\n    cuda_sources=r\'\'\''+cuda+"'''"+''',
    functions=["attention"],
    extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=False,
)
'''
a=s.index('\n_ROPE =') if '\n_ROPE =' in s else s.index('\n_ROPE:')
s=s[:a]+block+s[a:]
a=s.index('    plan = _plan(cfg, max_pages, device)',s.index('def kernel_fn'))
block='''    # The native single-tile grid wins where at most four KV heads fill one wave.
    # Keep the established Triton plans on larger batches and verify blocks.
    if (cfg["window"] == 2048 and length == 1 and batch * hkv <= 4
            and group == 8 and d == 64 and cfg["page_size"] in (1, 64) and cfg["rope_theta"]):
        splits = min(32, triton.cdiv(max_pages * cfg["page_size"], 64))
        pacc = torch.empty((batch, hkv, splits, 8, 64), dtype=torch.float32, device=device)
        pml = torch.empty((batch, hkv, splits, 8, 2), dtype=torch.float32, device=device)
        out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
        cs = _cos_sin(cfg["rope_theta"], 64, device)
        _NATIVE_SLIDING.attention(qkv, cs, data["k_cache"], data["v_cache"], page_table,
                                 data["seq_lens"], data["cache_loc"], data["k_scale"],
                                 data["v_scale"], pacc, pml, cfg["page_size"],
                                 cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, True)
        _combine_hb[(batch, hq // 2)](
            qkv, pacc, pml, data["v_scale"], out,
            HQ=hq, HKV=hkv, G=8, L=1, M=8, D=64, S=splits,
            S_PAD=triton.next_power_of_2(splits), SB=triton.next_power_of_2(splits),
            HB=2, PDL=USE_PDL, num_warps=1, launch_pdl=USE_PDL)
        return out
'''
s=s[:a]+block+s[a:];p.write_text(s)
