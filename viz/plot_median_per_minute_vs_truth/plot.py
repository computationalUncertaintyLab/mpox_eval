#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

def mm2inch(x):
    return x/25.4

if __name__ == "__main__":

    d = pd.read_csv("../../models/individual_hj_at_minute_horizons/all_quantiles.csv")
    medians = d.loc[d["quantile"]==0.50]


    fromQid2Labels = { 11039:"Ttl cases in Canada"
                      ,10981:"Num of US states"
                      ,10979:"Ttl cases in US"
                      ,10978:"Ttl cases in Europe"
                      ,10975:"Num of Countries"
                      ,10977:"PHEIC"}

    
    fromQid2bounds = { 11039:[100,500]
                      ,10981:[10,40]
                      ,10979:[0,500]
                      ,10978:[1000,5500]
                      ,10975:[10,100]
                      ,10977:[0,0]
                       }
    
    truths = pd.read_csv("../../truth_tables/truths.csv")

    plt.style.use("fivethirtyeight")
    fig,axs = plt.subplots(3,2)

    axs = axs.flatten()
    for n, (qid,subset) in enumerate( medians.groupby(["question_id"]) ):
        ax=axs[n]

        sns.lineplot(x="horizon", y="value", data = subset,ax=ax, lw=2)
        sns.lineplot(x="horizon", y="value",hue="user_id",color="blue",alpha=0.25,lw=0.50, data = subset,ax=ax)

        ax.set_ylabel("{:s}".format(fromQid2Labels[qid]), fontsize=8)
        ax.set_xlabel("Forecast horizon", fontsize=8)

        truth = float(truths.loc[truths.question_id==qid,"truth"])

        ax.axhline(truth, lw=1, color="black")

        ax.get_legend().remove()
        ax.tick_params(which="both",labelsize=8)

        ax.invert_xaxis()

        lower,upper = fromQid2bounds[qid]
        ax.set_ylim(lower,upper)


        #xticks = np.arange(100,600+100,100)
        
        #tick_data  = subset.iloc[ xticks,:]["cut_point"]

        #ax.set_xticks(xticks)
        #ax.set_xticklabels( [ datetime.strptime(x,"%Y-%m-%d %H:%M:%S").strftime("%m/%d") for x in tick_data.values[::-1]]
        #                    ,fontsize=8  )

        
    axs[-1].set_xticks([])
    axs[-1].set_yticks([])
    
    w = mm2inch(183)
    fig.set_size_inches(w,w/1.5)

    fig.set_tight_layout(True)

    plt.savefig("medians_over_time.pdf")
    plt.savefig("medians_over_time.png", dpi=350)
    
    plt.close()





    

