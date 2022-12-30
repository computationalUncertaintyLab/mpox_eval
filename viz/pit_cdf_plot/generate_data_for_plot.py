#mcandrew

import sys
import numpy as np
import pandas as pd

def vals2cdf(x):
    x,L = sorted(x), len(x)
    return x,np.arange(1.,L+1)/L

if __name__ == "__main__":

    horizons = pd.read_csv("../../scores/individuals/PITS.csv")
   
    cdfs = {"pit":[],"prob":[],"qid":[],"horizon":[]}
    for (qid,horizon),x in horizons.groupby(["question_id","horizon"]):
        y,py = vals2cdf(x.PIT)

        N = len(y)
        cdfs["pit"].extend(y)
        cdfs["prob"].extend(py)
        cdfs["qid"].extend([qid]*N)
        cdfs["horizon"].extend([horizon]*N)
    cdfs = pd.DataFrame(cdfs)
    
    cdfs.to_csv("cdfs.csv",index=False)
