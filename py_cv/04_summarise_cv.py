#!/usr/bin/env python3
"""Paired held-out ELPD: Basic+z vs ESaDDM+z; subject-level uncertainty."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

KEY=['sub_id','phase','trial','subj_idx','response','rt']
def run(root,out,study,bootstrap=5000,seed=20261023):
 root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
 allparts=[]
 for f in range(5):
  dfs=[]
  for model in ['basic','contrast']:
   p=root/f'fold_{f}'/model/'score'/'heldout_pointwise.csv'
   flag=p.parent/'score_complete.flag'
   if not p.exists() or not flag.exists(): raise FileNotFoundError('Missing complete score: '+str(p))
   df=pd.read_csv(p)
   if set(df['study'])!={study} or set(df['fold'])!={f} or set(df['model'])!={model}:
    raise ValueError('Wrong study,fold,model in score '+str(p))
   if df.duplicated(KEY).any():raise ValueError('Duplicate scored trial in '+str(p))
   dfs.append(df[KEY+['heldout_lpd']].rename(columns={'heldout_lpd':f'lpd_{model}'}))
  x=dfs[0].merge(dfs[1],on=KEY,validate='one_to_one',how='outer',indicator=True)
  if not x._merge.eq('both').all():raise ValueError('Basic/contrast do not score same heldout trials')
  x=x.drop(columns='_merge');x.insert(0,'fold',f);x['difference_contrast_minus_basic']=x.lpd_contrast-x.lpd_basic
  allparts.append(x)
 full=pd.concat(allparts,ignore_index=True)
 if full.duplicated(KEY).any():raise ValueError('Heldout trial scored in more than one fold')
 manifest=json.loads((root/'fold_manifest.json').read_text())
 if len(full)!=manifest['total_rows']:raise ValueError('Scores do not cover every original trial exactly once')
 assign=pd.read_csv(root/'fold_assignments.csv')
 chk=full.merge(assign[['sub_id','phase','trial','subj_idx','cv_fold']],on=['sub_id','phase','trial','subj_idx'],validate='one_to_one',how='outer',indicator=True)
 if not chk._merge.eq('both').all() or not chk.fold.eq(chk.cv_fold).all():raise ValueError('Scores mismatch saved fold assignment')
 if not np.isfinite(full[['lpd_basic','lpd_contrast']].to_numpy()).all():raise ValueError('Non-finite lpd')
 full.to_csv(out/'paired_heldout_pointwise.csv',index=False)
 subj=full.groupby('subj_idx').agg(n_trials=('fold','size'),elpd_basic=('lpd_basic','sum'),elpd_contrast=('lpd_contrast','sum'),delta_elpd=('difference_contrast_minus_basic','sum'),mean_delta_lpd=('difference_contrast_minus_basic','mean')).reset_index()
 subj.to_csv(out/'participant_elpd.csv',index=False)
 fold=full.groupby('fold').agg(n_trials=('fold','size'),elpd_basic=('lpd_basic','sum'),elpd_contrast=('lpd_contrast','sum'),delta_elpd=('difference_contrast_minus_basic','sum')).reset_index()
 fold.to_csv(out/'fold_elpd.csv',index=False)
 # Participant cluster bootstrap preserves within-participant trial dependence;
 # statistic = total delta ELPD / total trials of resampled subjects.
 rng=np.random.RandomState(seed);numbers=subj.n_trials.to_numpy(float);deltas=subj.delta_elpd.to_numpy(float)
 b=np.empty(bootstrap)
 for i in range(bootstrap):
  ix=rng.randint(0,len(subj),len(subj));b[i]=deltas[ix].sum()/numbers[ix].sum()
 avg=float(full.difference_contrast_minus_basic.mean())
 outdict={'study':study,'n_participants':len(subj),'n_trials':len(full),'n_folds':5,
  'basic_elpd':float(full.lpd_basic.sum()),'contrast_elpd':float(full.lpd_contrast.sum()),
  'delta_elpd_contrast_minus_basic':float(full.difference_contrast_minus_basic.sum()),
  'delta_elpd_per_trial':avg,
  'participant_cluster_bootstrap_delta_per_trial_CI95':[float(x) for x in np.quantile(b,[.025,.975])],
  'participants_with_positive_delta':int((subj.delta_elpd>0).sum()),
  'note':'Explicit grouped 5-fold conditional heldout choice-and-RT predictive log density; uncertainty resamples participants; no CV tuning on outcomes.'}
 (out/'summary.json').write_text(json.dumps(outdict,indent=2))
 print(json.dumps(outdict,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--study',choices=['study1','study2'],required=True)
 a=p.parse_args();run(a.root,a.out,a.study)
