"""Experimental native merge only; attention/plans remain production's."""
import submission as sub
from experiments.verify_merge.native import ext
class Merge:
 def __init__(self,striped=False,pdl=True):self.striped=striped;self.pdl=pdl
 def __getitem__(self,grid):
  def call(qkv,pa,pm,vs,out,**kw):
   ext.merge(qkv,pa,pm,vs,out,kw['HQ'],kw['HKV'],kw['L'],kw['S'],self.striped,self.pdl)
  return call

def run(data,striped=False,pdl=True):
 cfg=data['config'];plan=sub._plan(cfg,data['page_table'].shape[1],data['qkv'].device)
 if not plan.get('gluon'):return sub.kernel_fn(data)
 original=sub._combine;sub._combine=Merge(striped,pdl)
 try:return sub.kernel_fn(data)
 finally:sub._combine=original
