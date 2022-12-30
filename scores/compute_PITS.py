#mcandrew

import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":

    hj_ensemble_preds = pd.read_csv("../models/equally_weighted_hj_ensemble/all_predictions.csv")
    truth = pd.read_csv("../truth_tables/truths.csv")

    hj_ensemble_preds = hj_ensemble_preds.merge(truth, on = ["question_id"])

    def PIT(x):
        from scipy.interpolate import interp1d
        from scipy.integrate import quad
        
        xs = x.original_value.values
        ys = x.value.values
        f = interp1d(xs,ys)

        t    = x.iloc[0]["truth"]
        minx = x.iloc[0]["min"]

        PIT = quad(f,minx,t)[0]
        return PIT

    PITS = hj_ensemble_preds.groupby(["question_id","horizon"]).apply(PIT)
