#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def mm2inch(x):
    return x/25.4

if __name__ == "__main__":

    cdfs = pd.read_csv("./cdfs.csv")
    
    fromQid2Labels = { 11039:"Ttl cases in Canada"
                      ,10981:"Num of US states"
                      ,10979:"Ttl cases in US"
                      ,10978:"Ttl cases in Europe"
                      ,10975:"Num of Countries"
                      ,10977:"PHEIC"}

    cdfs["qid"] = [ fromQid2Labels[x] for x in cdfs.qid.values]
    cdfs["horizon"] = ["{:d} wk ahead".format(x) if x==1 else "{:d} wks ahead".format(x) for x in cdfs.horizon.values ]

    cdfs = cdfs.rename(columns = {"qid":"Question", "horizon":"Forecast horizon"})
    
    plt.style.use("fivethirtyeight")

    fig,ax = plt.subplots()
    
    sns.lineplot( x="pit", y = "prob", hue = "Question", style="Forecast horizon", data = cdfs, linewidth=1 )

    ax.set_xticks(np.arange(0,1+0.25,0.25))
    ax.set_yticks([0,0.25,0.50,0.75,1.0])
        
    ax.set_ylabel("P(Y < PIT)", fontsize=10)
    ax.set_xlabel("Prob. Int. Transform (PIT)", fontsize=10)

    ax.legend( frameon=False, fontsize=10 )
    
    ax.tick_params(which="both", labelsize=8)

    fig.set_tight_layout(True)

    w = mm2inch(183)
    fig.set_size_inches(w,w/1.5)
    
    plt.savefig("cdfs_of_pits.pdf")
    plt.close()

