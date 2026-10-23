#!/usr/bin/env python3
"""Fit exactly the existing ES S-upper HDDMRegressor on a training fold."""
import argparse, importlib.util,json,sys
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('train','source','original-fit-script','out'): p.add_argument('--'+name,required=True,type=Path)
 p.add_argument('--study',choices=['study1','study2'],required=True)
 p.add_argument('--model',choices=['basic','contrast'],required=True)
 p.add_argument('--fold',type=int,required=True)
 p.add_argument('--chains',type=int,default=3);p.add_argument('--samples',type=int,default=4000)
 p.add_argument('--burn',type=int,default=1000);p.add_argument('--jobs',type=int,default=3);p.add_argument('--seed',type=int,default=20261023)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if a.burn>=a.samples:raise ValueError('burn >= samples')
 if a.study=='study1' and 'study1_fit' not in str(a.original_fit_script):raise ValueError('Wrong study fit script')
 if a.study=='study2' and 'study2_fit' not in str(a.original_fit_script):raise ValueError('Wrong study fit script')
 spec=importlib.util.spec_from_file_location('existing_es_fit',str(a.original_fit_script));fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)
 data=pd.read_csv(a.train,low_memory=False)
 audit,part,corr=fit.validate(data,a.source,a.model,True)
 predictors,formula,include,model_name=fit.spec(a.model,True)
 assert formula.startswith('v ~ 0 + ') and include==['a','t','v','z']
 expected='aDDM_ES_IDENTITY_S_UPPER_'+a.model.upper()+'_Z_NO_INTERCEPT'
 assert model_name==expected
 for chain in range(a.chains):
  if (a.out/f'{model_name}_{chain}.hddm').exists() or (a.out/f'{model_name}_{chain}.pkl').exists():
   raise FileExistsError('Refusing overwrite of pre-existing fold fit in '+str(a.out))
 (a.out/'fit_audit.json').write_text(json.dumps({'study':a.study,'fold':a.fold,'fit_audit':audit,'chains':a.chains,'samples':a.samples,'burn':a.burn,'seed':a.seed,'n_train_rows':len(data)},indent=2))
 print(json.dumps(audit,indent=2),flush=True)
 Parallel(n_jobs=min(a.jobs,a.chains))(
  delayed(fit.fit_chain)(c,data,a.out,a.model,True,model_name,a.samples,a.burn,a.seed+1000*a.fold+(0 if a.model=='basic' else 100)) for c in range(a.chains)
 )
 (a.out/'fit_complete.flag').write_text('All expected chains saved\n')
 print('COMPLETE',a.study,a.fold,a.model,flush=True)
