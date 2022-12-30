#mcandrew

import sys
import numpy as np
import pandas as pd

import joblib
from joblib import dump, load
from joblib import Parallel, delayed

if __name__ == "__main__":

    continuous_predictions_at_all_horizons = pd.read_csv("../../raw_data/human_judgment_predictions/continuous_all_horizons_predictions_at_hours_cuts.csv")
    continuous_predictions_at_all_horizons = continuous_predictions_at_all_horizons.loc[~continuous_predictions_at_all_horizons.variable.isin(["P(r<0)","P(r>1)"])]
    

    #--normalize the abve to one
    def norm(x):
        from scipy.interpolate import interp1d
        from scipy.integrate import quad

        xs = x.original_value.values
        ys = x.value.values

        f = interp1d(xs,ys)

        I = quad(f,min(xs),max(xs))[0]

        x["value"] = x.value.values/I
        
        return x

    for n,(idx, x) in enumerate(continuous_predictions_at_all_horizons.groupby(["question_id","user_id","resolve_time","resolution","cut_point","min","max","horizon"])):
        if len(x) < 100: #--there are some odd anomolies where users have a density with less than 101 interplation points. These user densities are removed. 
            continue
        
        x = norm(x)
        
        if n==0:
            x.to_csv("all_predictions.csv",index=False,mode="w",header=True) 
        else:
            x.to_csv("all_predictions.csv",index=False,mode="a",header=False)
    
    continuous = pd.read_csv("./all_predictions.csv")

    #--generate quantiles
    def build_quantiles(n,idx_x):
        from scipy.interpolate import interp1d
        from scipy.integrate import quad
        from scipy.optimize import root_scalar as root

        idx,x = idx_x
        
        xs = x.original_value.values
        ys = x.value.values

        f = interp1d(xs,ys)

        F_at_xs = np.array([ quad(f,min(xs),x)[0] for x in xs ])

        F = interp1d(xs,F_at_xs)
        
        qs = {"quantile":[],"value":[]}
        for q in np.arange(0.01,0.99+0.01,0.01):

            lower = np.where(F_at_xs<q)[0][-1]
            upper = lower+1
            
            try:
                val = root( lambda x: F(x)-q, bracket = [xs[lower], xs[upper]]).root
            except:
                val = np.nan
                
            qs["quantile"].append(q)
            qs["value"].append(val)

        qs = pd.DataFrame(qs)
        for i,label in zip(idx,["question_id","user_id","resolve_time","resolution","cut_point","min","max","horizon"]):
            qs[label] = i

        if n==0:
            qs.to_csv("all_quantiles.csv",index=False,mode="w",header=True) 
        else:
            qs.to_csv("all_quantiles.csv",index=False,mode="a",header=False)
            
        return 0

    Parallel(n_jobs=28)(
        delayed(build_quantiles)(n,idx_x) for n,idx_x in enumerate(continuous.groupby(["question_id","user_id","resolve_time","resolution","cut_point","min","max","horizon"])))
