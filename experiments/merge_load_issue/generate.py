"""Issue both independent partial arrays before consuming either result.

Only the native full merger changes. Every read stays after the PDL wait.
The arithmetic, layouts, grid and precision stay identical to the parent.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def generate(source):
    source = source.replace('name="muse_native_merge"',
                            'name="muse_native_merge_load_issue_s43"', 1)
    start = source.index(' float2 a[N];')
    end = source.index(' float x=0,y=0;', start)
    loads = source[start:end]
    source = source[:start] + source[end:]
    needle = '   m=fmaxf(m,ml[i].x);\n }\n m=max32(m);'
    replacement = (' }\n' + loads +
                   ' #pragma unroll\n'
                   ' for(int i=0;i<NS;i++)m=fmaxf(m,ml[i].x);\n'
                   ' m=max32(m);')
    assert source.count(needle) == 1
    return source.replace(needle, replacement, 1)

if __name__ == '__main__':
    (ROOT/'candidate.py').write_text(generate((ROOT/'previous.py').read_text()))
