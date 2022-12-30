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

    wide = pd.pivot_table(index="model",columns=["question_id","horizon"],values=["WIS"], data = d)

    #--subset to the endembles
    wide = wide.loc[wide.index.isin(["ensemble_hj","ensemble_comp"])]

    relative_wis = wide.iloc[0,:] / wide.iloc[1,:]
    relative_wis = relative_wis.reset_index()

    #--long
    longdata = d.loc[d.model.isin(["ensemble_hj","ensemble_comp"])]
    
    hue_order = ["LL","hj","wag"]
    
    plt.style.use("fivethirtyeight")
    
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 2)

    ax = fig.add_subplot(gs[0, 0])

    #--two options
    longdata["model"] = ["Human Judgment" if x=="ensemble_hj" else "Computational models" for x in longdata.model]
    g = sns.boxplot(  x="horizon"  ,y="WIS",hue="model", data= longdata, boxprops=dict(alpha=.3,lw=1), fliersize=0,linewidth=1,ax=ax)

    for horizon, data in d.groupby(["horizon"]):
        comp_model = data.loc[data.model=="ensemble_comp"]
        hj_model   = data.loc[data.model=="ensemble_hj"] 

        N,M = len(comp_model), len(hj_model)

        jitter_comp  = np.random.normal(0,0.005,N)
        jitter_human = np.random.normal(0,0.005,M)

        ax.scatter( x = (horizon-1+0.25) + jitter_comp , y=comp_model.WIS , alpha=0.80, color = "red" )
        ax.scatter( x = (horizon-1-0.25) + jitter_human, y=hj_model.WIS   , alpha=0.80, color = "blue" )

        x0,x1 = (horizon-1-0.25), (horizon-1+0.25) #--human and then comp
        for n,qid in enumerate(comp_model.question_id):

            y0 = float(hj_model.loc[hj_model.question_id==qid,"WIS"])
            y1 = float(comp_model.loc[comp_model.question_id==qid,"WIS"])
            ax.plot( [x0 + jitter_human[n],x1+jitter_comp[n]], [y0,y1], color="black",alpha=0.5,ls="--",lw=1)
    
    ax.tick_params(which="both",labelsize=8)

    ax.set_ylabel("WIS", fontsize=10)
    ax.set_xlabel("",fontsize=10)

    ax.set_xticklabels(["1 week","2 week","3 week"])
    
    handles, labeles = ax.get_legend_handles_labels()
    ax.legend(handles[:2+1],["Human judgment","Computational models"],fontsize=10,ncol=1,frameon=False,loc="upper left",bbox_to_anchor=(-.05,1.15))
    ax.text(0.0,0.920,s="A.",ha="left",va="top",fontweight="bold",fontsize=8,transform=ax.transAxes)
    
    d = longdata

    def cover(x):
        x = x[ [_ for _ in x.columns if "cover" in _]  ]
        x = x.mean(0)
        return x
    coverages = d.groupby(["model","horizon"]).apply(cover)

    def cover_nums(x):
        x = x[ [_ for _ in x.columns if "cover" in _]  ]
        x = x.sum(0)
        return x
    coverages_nums = d.groupby(["model"]).apply(cover_nums)
    
    Nquestions = len(d.question_id.unique())
    
    coverages = coverages.reset_index().melt(id_vars=["model","horizon"])
    coverages["cover"] = coverages.variable.str.replace("cover_","").astype(float)

    hue_order = ["ensemble_hj","ensemble_comp"]
    
    ax = fig.add_subplot(gs[0,1])
        
    sns.lineplot(x="cover", y="value", hue="model", style="horizon", hue_order=["Human Judgment","Computational models"], data = coverages ,ax=ax, lw=1.5)

    handles, labeles = ax.get_legend_handles_labels()

    #--label only the horizons
    leg = ax.legend( handles[-3:], ["1 wk ahead","2 wks ahead","3 wks ahead"], frameon=False,fontsize=10, bbox_to_anchor = (0.50,0.50))

    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    
    ax.plot([0.,1],[0.,1], color="black", ls="-", lw=1.5)
    
    ax.set_xlabel("Theoretical coverage", fontsize=10)

    ax.set_ylabel("Empirical coverage", fontsize=10)
    ax.set_ylabel("", fontsize=10)

    ax.set_xticks([0,0.25,0.50,0.75,1.0])
    ax.set_yticks([0,0.25,0.50,0.75,1.0])
            
    ax.tick_params(which="both",labelsize=8)

    ax.text(0.0,0.920,s="B.",ha="left",va="top",fontweight="bold",fontsize=8,transform=ax.transAxes)
    

    #-----------------------INDIVIDUALS BELOW---------------------------------------------
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
    
    #color = "blue"
    
    ax = fig.add_subplot(gs[1,0])
    sns.boxplot(  x="horizon"  ,y="relative_wis",  data= wide_individual_data, boxprops=dict(alpha=.6,lw=1), fliersize=0,ax=ax,linewidth=1, palette="colorblind")
    sns.stripplot(x="horizon"  ,y="relative_wis" , dodge=True, jitter=True,data= wide_individual_data,ax=ax, alpha = 0.2, s=3, palette="colorblind")

    ax.axhline(1.0,ls="--",lw=1,color="black")

    ax.tick_params(which="both",labelsize=8)

    ax.set_ylabel("Relative WIS\n(Ref. = Random Walk)", fontsize=10)
    ax.set_xlabel("",fontsize=10)

    ax.set_xticklabels(["1 week","2 week","3 week"])

    handles, labeles = ax.get_legend_handles_labels()
    ax.legend(handles[:2+1],["Local model","Human judgment","Exponential"],fontsize=10,ncol=2,frameon=False)

    ax.set_yticks([0,1,2,4,6,8])
    ax.set_ylim(-0.1,9)

    ax.text(0.0,0.920,s="C.",ha="left",va="top",fontweight="bold",fontsize=8,transform=ax.transAxes)
    
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
    
    ax = fig.add_subplot(gs[1,1])
    sns.lineplot(x="cover", y="value", hue="horizon", ax=ax, lw=1.5, data = coverages, palette="colorblind")

    handles, labeles = ax.get_legend_handles_labels()
    ax.legend(handles,["1 wk ahead","2 wks ahead","3 wks ahead"],frameon=False,fontsize=10)

    ax.plot([0,1],[0,1], color="black", ls="-", lw=1.5)
    
    ax.set_xlabel("Theoretical coverage", fontsize=10)

    ax.set_ylabel("Empirical coverage", fontsize=10)

    ax.set_xticks([0,0.25,0.50,0.75,1.0])
    ax.set_yticks([0,0.25,0.50,0.75,1.0])
            
    ax.tick_params(which="both",labelsize=8)

    ax.text(0.0,0.920,s="D.",ha="left",va="top",fontweight="bold",fontsize=8,transform=ax.transAxes)
    
    w = mm2inch(183)
    fig.set_size_inches(w,w*0.75)
        
    #fig.set_tight_layout(True)

    plt.subplots_adjust(wspace=0.0, hspace=0.01)
    
    plt.savefig("wis_and_coverage__combined.pdf")
    plt.savefig("wis_and_coverage__combined.png",dpi=350)
    
    plt.close()
