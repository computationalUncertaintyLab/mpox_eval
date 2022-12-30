#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def mm2inch(x):
    return x/25.4

from datetime import datetime, timedelta

if __name__ == "__main__":

    horizons = pd.read_csv("../../models/individual_hj_at_horizons/all_quantiles.csv")

    medians = horizons.loc[horizons["quantile"]==0.50]

    truths = pd.read_csv("../../truth_tables/truths.csv")

    medians = medians.merge(truths, on = ["question_id"])

    medians["ratio"] = medians["value"] / medians["truth"] - 1.


    def mean_and_95ci(x,y):
        m = x[y].mean()
        s = x[y].std()
        return pd.Series({"mean":np.round(m,2), "lower":np.round(m-1.96*s,2),"upper":np.round(m+1.96*s,2)})

    results = medians.groupby(["horizon"]).apply(lambda x: mean_and_95ci(x,"ratio"))
    print(results)
    
    

    
