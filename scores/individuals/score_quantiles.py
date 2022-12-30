#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("../")
from scoring import *

if __name__ == "__main__":

    #--combine all forecasts
    individual_quantiles__HJ = pd.read_csv("../../models/individual_hj_at_horizons/all_quantiles.csv")
    individual_quantiles__HJ["model"] = "individuals"

    #--extract quantiles
    quantiles = list( np.round(individual_quantiles__HJ["quantile"].unique(),3))
   
    all_quantiles = individual_quantiles__HJ

    #--format quantiles
    all_quantiles["quantile"] = np.round(all_quantiles["quantile"],3)

    #--subset to have the same quantiles
    all_quantiles = all_quantiles.loc[all_quantiles["quantile"].isin(quantiles)]
    
    #--read in truth
    truth = pd.read_csv("../../truth_tables/truths.csv")

    all_quantiles = all_quantiles.merge(truth, on = ["question_id"])

    data = {"question_id":[],"user_id":[],"horizon":[],"model":[], "WIS":[], "MAE":[]}

    for alpha in np.round(np.arange( 0.01, 0.40, 0.01 ),3):
        data["cover_{:.2f}".format( 1-2*alpha )] = []
    
    for (qid,user_id,horizon,model), subset in all_quantiles.groupby(["question_id","user_id","horizon","model"]):

        truth = subset.iloc[0]["truth"]

        wis = WIS(subset, truth)

        data["question_id"].append(qid)
        data["user_id"].append(user_id)
        data["horizon"].append(horizon)
        data["model"].append(model)
        data["WIS"].append(wis)

        median = float(subset.loc[subset["quantile"]==0.50, "value"])
        data["MAE"].append( abs(median-truth)  )

        #--coverages
        for alpha in np.round(np.arange( 0.01, 0.40, 0.01 ),3):
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

    data.to_csv("./WIS_scores.csv", index=False)
