#mcandrew

import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":

    cont = pd.read_csv("../models/equally_weighted_hj_ensemble/all_predictions.csv")
    cont = cont[["question_id","horizon","cut_point","resolve_time"]]
    cont = cont.drop_duplicates()
    
    bina = pd.read_csv("../models/equally_weighted_hj_ensemble/binary_predictions.csv")
    bina = bina[["question_id","horizon","cut_point","resolve_time"]]
    bina = bina.drop_duplicates()

    all_data = cont.append(bina)

    all_data.to_csv("./cut_points_resolve_times_qids.csv", index=False)
