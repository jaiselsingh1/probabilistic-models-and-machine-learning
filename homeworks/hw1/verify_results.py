"""Independent recomputation from saved outputs, without importing fitting code."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, skewnorm
from scipy.optimize import brentq

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    out = args.out_dir
    required = ("results.json", "cohort.csv", "heldout_scores.csv", "comparisons.csv")
    missing = [name for name in required if not (out / name).exists()]
    if missing:
        parser.error("Missing " + ", ".join(missing) + "; run python -m homeworks.hw1.audited_analysis first.")
    meta=json.loads((out/'results.json').read_text())
    cohort=pd.read_csv(out/'cohort.csv').set_index('row_id')
    scores=pd.read_csv(out/'heldout_scores.csv')
    comparisons=pd.read_csv(out/'comparisons.csv')
    maximum_error=0.0
    modes={}
    for variant, fit_groups in meta['fits'].items():
        held=scores.loc[scores.variant==variant]
        assert (cohort.loc[held.row_id,'partition']=='test').all()
        ref=None if variant=='raw' else pd.read_csv(out/f'reference_{variant}.csv')
        raw=cohort.loc[held.row_id]
        y=raw.raw_ratio.to_numpy(copy=True)
        if ref is not None:
            y-=np.interp(raw.ability,ref.iloc[:,0],ref.iloc[:,1])
        np.testing.assert_allclose(y,held.y,rtol=0,atol=1e-14)
        for group,fit in fit_groups.items():
            sub=held.loc[held.group==group]; yy=sub.y.to_numpy(); pars=fit['parameters']
            p=pars['mixture']; w=p['weight_upper']
            vals={
                'gaussian': norm.logpdf(yy,pars['gaussian']['mu'],pars['gaussian']['sigma']),
                'skewnormal':skewnorm.logpdf(yy,pars['skewnormal']['shape'],pars['skewnormal']['loc'],pars['skewnormal']['scale']),
                'mixture': np.log((1-w)*norm.pdf(yy,p['mu_lower'],p['sigma_lower'])+w*norm.pdf(yy,p['mu_upper'],p['sigma_upper']))
            }
            for model,value in vals.items():
                maximum_error=max(maximum_error,float(np.max(np.abs(value-sub[f'logp_{model}']))))
                np.testing.assert_allclose(value,sub[f'logp_{model}'],rtol=0,atol=1e-11)
            # Every stationary point is between the two component means: outside,
            # both normal derivatives have the same sign. Find sign changes there.
            def derivative(x):
                return ((1-w)*norm.pdf(x,p['mu_lower'],p['sigma_lower'])*(p['mu_lower']-x)/p['sigma_lower']**2
                        +w*norm.pdf(x,p['mu_upper'],p['sigma_upper'])*(p['mu_upper']-x)/p['sigma_upper']**2)
            grid=np.linspace(p['mu_lower'],p['mu_upper'],100001)
            d=derivative(grid); crossings=np.flatnonzero(d[:-1]*d[1:]<0)
            roots=[brentq(derivative,grid[i],grid[i+1]) for i in crossings]
            assert len(roots)==1 and derivative(roots[0]-1e-6)>0 and derivative(roots[0]+1e-6)<0
            modes[f'{variant}:{group}']=roots[0]
        for _,row in comparisons.loc[comparisons.variant==variant].iterrows():
            sub=held if row.group=='all' else held.loc[held.year==2014] if row.group=='2014 only' else held.loc[held.group==row.group]
            other=row.comparison.removeprefix('mixture_minus_')
            dd=sub.logp_mixture-sub[f'logp_{other}']
            se=np.sqrt(sum(len(group)*np.var(dd.loc[group.index],ddof=1) for _,group in sub.groupby('group')))
            np.testing.assert_allclose([dd.sum(),se],[row.gain_total_nats,row.se_total_nats],rtol=0,atol=1e-10)
    result={'status':'passed','pointwise_rows_checked':len(scores),'maximum_log_density_error':maximum_error,
            'comparison_rows_checked':len(comparisons),'all_sixteen_mixtures_have_one_derivative_root':True,
            'derivative_roots':modes,'scope':'Independent score/SE/response recomputation; numerical mode cross-check, not proof of global optimization.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='derivative_roots'},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
