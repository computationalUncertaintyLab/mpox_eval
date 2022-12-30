#mcandrew

import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":

    continuous_predictions_at_all_horizons = pd.read_csv("../../raw_data/human_judgment_predictions/continuous_all_horizons_predictions.csv")

    def generate_EW_ensemble(x):
        EW_probs = pd.pivot_table(index="original_value", columns = "user_id", values = "value", data = x ).mean(1).reset_index()
        EW_probs.columns = ["original_value","value"]
        return EW_probs
    continuous_EW_ensemble = continuous_predictions_at_all_horizons.groupby(["question_id","resolve_time","resolution","cut_point","min","max","horizon"]).apply(generate_EW_ensemble).reset_index()

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
    continuous_EW_ensemble = continuous_EW_ensemble.groupby(["question_id","resolve_time","resolution","cut_point","min","max","horizon"]).apply(norm).reset_index()

    continuous_EW_ensemble.to_csv("./all_predictions.csv", index=False)

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
    all_quantiles = continuous_EW_ensemble.groupby(["question_id","resolve_time","resolution","cut_point","min","max","horizon"]).apply(build_quantiles).reset_index()
    
    all_quantiles.to_csv("./all_quantiles.csv",index=False)
    
    #--binary ensemble

    binary_predictions_at_all_horizons = pd.read_csv("../../raw_data/human_judgment_predictions/binary_all_horizons_predictions.csv")
    binary_EW_ensemble = binary_predictions_at_all_horizons

    binary_EW_ensemble.to_csv("./binary_predictions.csv")

    def build_quantiles(x):
        from scipy.interpolate import interp1d
        from scipy.integrate import quad
        from scipy.optimize import root_scalar as root
        
        qs = {"quantile": list(np.arange(0.01,0.99+0.01,0.01))
              ,"value"  : np.percentile( x.binary_prediction, 100*np.arange(0.01,0.99+0.01,0.01) ) }
        return pd.DataFrame(qs)
    all_quantiles = binary_EW_ensemble.groupby(["question_id","resolve_time","resolution","cut_point","horizon"]).apply(build_quantiles).reset_index()
    
    all_quantiles.to_csv("./all_binary_quantiles.csv",index=False)
 
    
