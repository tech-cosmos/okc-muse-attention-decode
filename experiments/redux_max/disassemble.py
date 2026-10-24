"""Build/extract only native extension modules; archive disassembly without GPU work."""
import ast,subprocess
from pathlib import Path
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parent
for name in ('previous','candidate'):
 tree=ast.parse((ROOT/(name+'.py')).read_text())
 node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_MERGE' for t in n.targets))
 ns={'load_inline':load_inline};exec(compile(ast.Module(body=[node],type_ignores=[]),'native-extract','exec'),ns)
 for flag,suffix in (('--dump-sass','.sass'),('--dump-resource-usage','-resources.txt')):
  report=subprocess.check_output(['/usr/local/cuda-13.0/bin/cuobjdump',flag,ns['_NATIVE_MERGE'].__file__],text=True)
  (ROOT/(name+suffix)).write_text('\n'.join(line.rstrip() for line in report.rstrip().splitlines())+'\n')
