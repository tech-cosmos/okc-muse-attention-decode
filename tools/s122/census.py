import re, sys
lines = open(sys.argv[1]).read().splitlines()
lo, hi = int(sys.argv[2], 16), int(sys.argv[3], 16)
skip = [tuple(int(x, 16) for x in s.split('-')) for s in sys.argv[4:]]
from collections import Counter
c = Counter(); n = 0
for l in lines:
    m = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)(\.[A-Z0-9_.]*)?', l)
    if not m: continue
    a = int(m.group(1), 16)
    if not (lo <= a <= hi) or any(s <= a < e for s, e in skip): continue
    c[m.group(3)] += 1; n += 1
print(n, c.most_common(40))
