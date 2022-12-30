#mcandrew

import sys


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scoring import *

if __name__ == "__main__":

    #--combine all forecasts
    ensemble_quantiles__HJ = pd.read_csv("../models/equally_weighted_hj_ensemble/all_quantiles.csv")
    ensemble_quantiles__HJ["model"] = "ensemble_hj"

    wag_model_quantiles     = pd.read_csv("../models/wag/all_quantiles.csv")
    ll_model_quantiles      = pd.read_csv("../models/latent_local/all_quantiles.csv") 
    rw_model_quantiles      = pd.read_csv("../models/random_walk/all_quantiles.csv")
    comp_ensemble_quantiles = pd.read_csv("../models/computational_ensemble/all_quantiles.csv")
    
    #--convert percentiles to quantiles
    wag_model_quantiles["quantile"]      = wag_model_quantiles["quantile"]/100
    ll_model_quantiles["quantile"]       = ll_model_quantiles["quantile"]/100
    rw_model_quantiles["quantile"]       = rw_model_quantiles["quantile"]/100
    comp_ensemble_quantiles["quantile"]  = comp_ensemble_quantiles["quantile"]/100

    #--extract quantiles
    quantiles = list( np.round(ensemble_quantiles__HJ["quantile"].unique(),3))
    
    all_quantiles = ensemble_quantiles__HJ.append(wag_model_quantiles).append(ll_model_quantiles).append(rw_model_quantiles).append(comp_ensemble_quantiles)

    #--format quantiles
    all_quantiles["quantile"] = np.round(all_quantiles["quantile"],3)

    #--subset to have the same quantiles
    all_quantiles = all_quantiles.loc[all_quantiles["quantile"].isin(quantiles)]
    
    #--read in truth
    truth = pd.read_csv("../truth_tables/truths.csv")

    all_quantiles = all_quantiles.merge(truth, on = ["question_id"])

    data = {"question_id":[],"horizon":[],"model":[], "WIS":[], "MAE":[]}

    for alpha in np.round(np.arange( 0.01, 0.49+0.01, 0.01 ),3):
        data["cover_{:.2f}".format( 1-2*alpha )] = []
    
    for (qid,horizon,model), subset in all_quantiles.groupby(["question_id","horizon","model"]):
        truth = subset.iloc[0]["truth"]

        wis = WIS(subset, truth)

        data["question_id"].append(qid)
        data["horizon"].append(horizon)
        data["model"].append(model)
        data["WIS"].append(wis)

        median = float(subset.loc[subset["quantile"]==0.50, "value"])
        data["MAE"].append( abs(median-truth)  )

        #--coverages
        for alpha in np.round(np.arange( 0.01, 0.49+0.01, 0.01 ),3):
            one_minus_alpha = np.round(1-alpha,3)
            lower,upper = float(subset.loc[subset["quantile"]==alpha, "value"]), float(subset.loc[subset["quantile"]==one_minus_alpha, "value"])
            cover = 1 if lower < truth < upper else 0

            data["cover_{:.2f}".format( 1-2*alpha )].append(cover)
            
    data = pd.DataFrame(data)

    #--coverage data
    cover = data.loc[:, ["question_id","horizon","model"] + [x for x in data.columns if "cover" in x] ]
    cover_long = cover.melt(id_vars = ["question_id","horizon","model"])
    cover_long["coverage"] = cover_long["variable"].str.replace("cover_","").astype(float)
    
    covers = cover_long.groupby(["horizon","model","coverage"]).apply( lambda x: pd.Series( {"p": x.value.mean() } ))
    covers = covers.reset_index()

    #--build a relative WIS with WAG as the reference
    data__wide = pd.pivot_table( index=["question_id","horizon"], columns = ["model"], values = ["WIS"], data = data )

    data.to_csv("WIS_scores.csv", index=False)
