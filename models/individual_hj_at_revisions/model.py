#mcandrew

import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":

    continuous_predictions_at_all_horizons = pd.read_csv("../../raw_data/human_judgment_predictions/continuous_no_horizons_predictions.csv")
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
    #continuous = continuous_predictions_at_all_horizons.groupby(["question_id","user_id","resolve_time","resolution","cut_point","min","max","horizon"]).apply(norm).reset_index()

    for n,(idx, x) in enumerate(continuous_predictions_at_all_horizons.groupby(["question_id","user_id","resolve_time","resolution","min","max","time"])):
        x = norm(x)

        if n==0:
            x.to_csv("./all_predictions.csv",mode="w",header=True,index=False)
        else:
            x.to_csv("./all_predictions.csv",mode="a",header=False,index=False)
    #continuous.to_csv("./all_predictions.csv", index=False)

    #--generate quantiles
    def build_quantiles(x):
        from scipy.interpolate import interp1d
        from scipy.integrate import quad
        from scipy.optimize import root_scalar as root
        
        xs = x.original_value.values
        ys = x.value.values

        f = interp1d(xs,ys)

        F_at_xs = np.array([ quad(f,min(xs),x)[0] for x in xs ])

        F = interp1d(xs,F_at_xs)
        
        qs = {"quantile":[],"value":[]}
        for q in np.arange(0.01,0.99+0.01,0.01):

            lower = np.where(F_at_xs<q)[0][-1]
            upper = lower+1
            
            val = root( lambda x: F(x)-q, bracket = [xs[lower], xs[upper]]).root

            qs["quantile"].append(q)
            qs["value"].append(val)
        return pd.DataFrame(qs)

    continuous = pd.read_csv("./all_predictions.csv")
    
    all_quantiles = continuous.groupby(["question_id","user_id","resolve_time","resolution","min","max","time"]).apply(build_quantiles).reset_index()
    all_quantiles.to_csv("./all_quantiles.csv",index=False)
    

