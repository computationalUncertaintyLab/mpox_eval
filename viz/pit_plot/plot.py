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


    horizons = pd.read_csv("../../scores/individuals/PITS.csv")
    hours    = pd.read_csv("../../scores/minute_scores/PITS.csv")

    fromQid2Labels = { 11039:"Ttl cases in Canada"
                      ,10981:"Num of US states"
                      ,10979:"Ttl cases in US"
                      ,10978:"Ttl cases in Europe"
                      ,10975:"Num of Countries"
                      ,10977:"PHEIC"}
    
    plt.style.use("fivethirtyeight")

    fig,axs = plt.subplots(2,1)

    ax = axs[0]

    sns.boxplot(x="question_id"  ,y="PIT",hue="horizon", data = horizons, fliersize=0,boxprops=dict(alpha=.8,lw=1), linewidth=1,ax=ax)

    ax.set_xticklabels(  [fromQid2Labels[int(x.get_text())] for x in ax.get_xticklabels()], fontsize=10 )

    ax.set_xlabel("")
    ax.set_ylabel("Prob. Int. Transform (PIT)", fontsize=10)

    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles,["1 wk ahead", "2 wks", "3 wks"], frameon=False, ncol=3, fontsize=10)

    ax.set_yticks([0,0.25,0.50,0.75,1.0])

    ax.tick_params(which="both", labelsize=8)

    
    #--PITs over time
    ax = axs[1]
    sns.lineplot( x="horizon", y="PIT", hue = "question_id", data = hours,ax=ax, linewidth=2., palette = "colorblind")

    
    ax.tick_params(which="both",labelsize=8)

    ax.set_xlabel("Forecast horizon", fontsize=8)

    ax.set_ylabel("Prob. Int. Transform (PIT)", fontsize=10)

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles, [ fromQid2Labels[int(x)] for x in labels]
              ,  ncol=3
              , frameon=False
              , fontsize=10)

    xticks = np.arange(100,600+100,100)
    
    time_data = pd.read_csv("../../models/individual_hj_at_minute_horizons/all_quantiles.csv")
    time_data = time_data[["horizon","question_id","cut_point"]].drop_duplicates()
    
    tick_data  = time_data.iloc[ xticks,:]["cut_point"]
    
    ax.invert_xaxis()


    ax.set_xticks( [1, 2*24, 24*7, 2*24*7, 3*24*7] )
    ax.set_xticklabels(["1 hour", "2 days","1 week before ground truth","2 weeks","3 weeks"])

    ax.set_ylim(0.3,1.1)
    ax.set_yticks([0.4,0.50,0.60,0.75,1.0])
    
    w = mm2inch(183)
    fig.set_size_inches(w,w/1.5)

    fig.set_tight_layout(True)

    plt.savefig("pits_over_time.pdf")
    plt.savefig("pits_over_time.png", dpi=350)
    
    plt.close()
