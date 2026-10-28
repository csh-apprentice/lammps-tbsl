#!/usr/bin/env python3
import csv,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def table(path):
 with path.open() as stream:
  reader=csv.DictReader(stream);rows=list(reader)
 return {name:np.asarray([float(row[name]) for row in rows]) for name in rows[0]}

def frame(paths):
 records=[]
 for path in paths:
  lines=path.read_text().splitlines();i=next(i for i,line in enumerate(lines) if line.startswith('ITEM: ATOMS'))
  names=lines[i].split()[2:];records.extend([list(map(float,line.split())) for line in lines[i+1:] if line])
 data=np.asarray(records);data=data[np.argsort(data[:,0])]
 return {name:data[:,i] for i,name in enumerate(names)}

results={}
for ranks in [1,2,4]:
 directory=HERE/'results'/f'mpi{ranks}';ref=table(directory/'reference.csv');cases={}
 original=frame(sorted(directory.glob('checkpoint_*.2.dump')))
 for mode in ['guarded','unguarded']:
  final=table(directory/f'{mode}_final.csv');initial=frame(sorted(directory.glob(mode+'_*.2.dump')))
  cases[mode]={'all_final_fields_exact':all(np.array_equal(ref[name],final[name]) for name in ref),'max_abs_final_differences':{name:float(np.max(np.abs(ref[name]-final[name]))) for name in ref},'charge_at_checkpoint_after_setup':initial['q'].tolist(),'charge_at_checkpoint_reference':original['q'].tolist(),'force_at_checkpoint_exact':all(np.array_equal(initial[name],original[name]) for name in ['fx','fy','fz'])}
 results[str(ranks)]=cases
(HERE/'results/comparison.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
