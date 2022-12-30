#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

def mm2inch(x):
    return x/25.4

if __name__ == "__main__":

    d = pd.read_csv("../../scores/WIS_scores.csv")
    d_individual = pd.read_csv("../../scores/individuals/WIS_scores.csv")
    d_individual["model"] = d_individual["user_id"]
    
    wide            = pd.pivot_table(index="model",columns=["question_id","horizon"],values=["WIS"], data = d)
    wide_individual = pd.pivot_table(index="model",columns=["question_id","horizon"],values=["WIS"], data = d_individual)

    wide_individual_data = pd.DataFrame()
    for user_id,row in wide_individual.iterrows():

        row = row/wide.iloc[-1]
        
        row = row.reset_index()
        row = row.drop(columns = ["level_0"])

        row.columns = ["question_id","horizon","relative_wis"]
        row["model"] = user_id

        wide_individual_data = wide_individual_data.append(row)
    wide_individual_data = wide_individual_data.reset_index()
    

    LL  = wide.iloc[0]  / wide.iloc[-1]
    wag = wide.iloc[1]  / wide.iloc[-1]
    hj  = wide.iloc[2]  / wide.iloc[-1]

    LL = LL.reset_index()
    LL = LL.rename(columns = {0:"relative_wis"})
    LL = LL[["question_id","horizon","relative_wis"]]
    LL["model"] = "LL"

    hj = hj.reset_index()
    hj = hj.rename(columns = {0:"relative_wis"})
    hj = hj[["question_id","horizon","relative_wis"]]
    hj["model"] = "hj"

    wag = wag.reset_index()
    wag = wag.rename(columns = {0:"relative_wis"})
    wag = wag[["question_id","horizon","relative_wis"]]
    wag["model"] = "wag"

    relative_wis = LL.append(hj).append(wag)
    relative_wis = relative_wis.reset_index()

    hue_order = ["LL","hj","wag"]
    
    plt.style.use("fivethirtyeight")
    
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 3)

    color = "blue"
    
    ax = fig.add_subplot(gs[0, :])
    sns.boxplot(  x="horizon"  ,y="relative_wis",  data= wide_individual_data, color = color, boxprops=dict(alpha=.6,lw=1), fliersize=0,ax=ax,linewidth=1)
    sns.stripplot(x="horizon"  ,y="relative_wis" , dodge=True, color = "black", jitter=True,data= wide_individual_data,ax=ax, alpha = 0.2, s=3)

    ax.axhline(1.0,ls="--",lw=1,color="black")

    ax.tick_params(which="both",labelsize=8)

    ax.set_ylabel("Relative WIS\n(Ref. = Random Walk)", fontsize=10)
    ax.set_xlabel("",fontsize=10)

    ax.set_xticklabels(["1 week","2 week","3 week"])

    handles, labeles = ax.get_legend_handles_labels()
    ax.legend(handles[:2+1],["Local model","Human judgment","Exponential"],fontsize=10,ncol=2,frameon=False)

    ax.set_yticks([0,1,2,4,6,8])
    ax.set_ylim(-0.1,9)

    
    def cover(x):
        x = x[ [_ for _ in x.columns if "cover" in _]  ]
        x = x.mean(0)
        return x
    coverages = d_individual.groupby(["model","horizon"]).apply(cover)

    def cover_nums(x):
        x = x[ [_ for _ in x.columns if "cover" in _]  ]
        x = x.sum(0)
        return x
    coverages_nums = d_individual.groupby(["model"]).apply(cover_nums)
    
    Nquestions = len(d_individual.question_id.unique())
    
    coverages = coverages.reset_index().melt(id_vars=["model","horizon"])
    coverages["cover"] = coverages.variable.str.replace("cover_","").astype(float)

    hue_order = ["LL","ensembleHJ","WAG","random"]
    
    for n,horizon in enumerate(np.arange(1,3+1)):
        ax = fig.add_subplot(gs[1,n])
        
        sns.lineplot(x="cover", y="value", data = coverages.loc[coverages.horizon==horizon] ,ax=ax, lw=1.5)

        handles, labeles = ax.get_legend_handles_labels()

        #if n==0:
        #    ax.legend([handles[-1]],["Random walk"],fontsize=10,ncol=1,frameon=False,loc="lower center")
        #else:
        #    ax.get_legend().remove()
        
        ax.plot([0.2,1],[0.2,1], color="black", ls="--", lw=1.5)
    
        ax.set_xlabel("Theoretical coverage", fontsize=10)

        if n ==0:
            ax.set_ylabel("Empirical coverage", fontsize=10)
        else:
            ax.set_ylabel("", fontsize=10)

        ax.set_xticks([0,0.25,0.50,0.75,1.0])
        ax.set_yticks([0,0.25,0.50,0.75,1.0])
            
        ax.tick_params(which="both",labelsize=8)

    w = mm2inch(183)
    fig.set_size_inches(w,w*0.75)
        
    fig.set_tight_layout(True)

    plt.savefig("wis_and_coverage__individuals.pdf")
    plt.savefig("wis_and_coverage__individuals.png",dpi=350)
    
    plt.close()


    #--proportion below random walk
    h3 = wide_individual_data.loc[wide_individual_data.horizon==3]
    np.mean(h3.relative_wis < 1.)
    
    h2 = wide_individual_data.loc[wide_individual_data.horizon==2]
    np.mean(h2.relative_wis < 1.)

    h1 = wide_individual_data.loc[wide_individual_data.horizon==1]
    np.mean(h1.relative_wis < 1.)
