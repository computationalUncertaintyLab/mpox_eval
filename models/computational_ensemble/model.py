#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    #--all predicitons
    wag_predictions = pd.read_csv("../wag/all_samples.csv")
    rw_predictins   = pd.read_csv("../random_walk/all_samples.csv")
    ll_predictions  = pd.read_csv("../latent_local/all_samples.csv")

    all_predictions = wag_predictions.append(rw_predictins).append(ll_predictions)

    all_predictions.to_csv("./all_predictions.csv",index=False)
    
    #--construct quantiles from the samples
    def build_quantiles(d):
        quantiles = list(100*np.arange(0.01,0.99+0.01,0.01))
        quantiles_values = np.percentile( d.value, quantiles )

        return pd.DataFrame({"quantile":quantiles, "value":quantiles_values})
    all_quantiles = all_predictions.groupby(["target","question_id","horizon","cut_point","resolve_time"]).apply(build_quantiles)
    all_quantiles = all_quantiles.reset_index()
    
    all_quantiles["model"] = "ensemble_comp"
    all_quantiles.to_csv("all_quantiles.csv",index=False)
