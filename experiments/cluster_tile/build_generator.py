from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'inherited_generate.py').read_text().replace('for bn in (128,256):','for bn in (256,):')
a=s.index("    s=s.replace('base=max(");z=s.index("    s=s.replace('for(int j=0;j<4",a)
s=s[:a]+'''    s=s.replace('base=max(0,seq-2047),lo=base+split*64','base=max(0,seq-2047),lo=base+split*'+str(bn))
    s=s.replace('j<2;j++',f'j<{bn//32};j++')
    s=s.replace('seq<lo+64',f'seq<lo+{bn}')
'''+s[z:]
s=s.replace('Build full L1 single-tile native kernels','Build sliding L1 single-tile native kernels')
s=s.replace('# Full attention does not use rope; keep binding shape for direct comparison.', '# Preserve the sliding RoPE, window and binding semantics.')
(p/'generate.py').write_text(s)
