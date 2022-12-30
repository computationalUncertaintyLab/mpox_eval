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

    d = pd.read_csv("../WIS_scores_for_all_models.csv")

    wide = pd.pivot_table(index="model",columns=["question_id","horizon"],values=["WIS"], data = d)

    #--subset to the endembles
    wide = wide.loc[wide.index.isin(["ensembleHJ","ensemble_comp"])]

    relative_wis = wide.iloc[0,:] / wide.iloc[1,:]
    relative_wis = relative_wis.reset_index()

    #--long
    longdata = d.loc[d.model.isin(["ensembleHJ","ensemble_comp"])]
    
    hue_order = ["LL","hj","wag"]
    
    plt.style.use("fivethirtyeight")
    
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 3)

    ax = fig.add_subplot(gs[0, :])

    #--two options
    longdata["model"] = ["Human Judgment" if x=="ensembleHJ" else "Computational models" for x in longdata.model]
    g = sns.boxplot(  x="horizon"  ,y="WIS",hue="model", data= longdata, boxprops=dict(alpha=.3,lw=1), fliersize=0,linewidth=1,ax=ax)

    for horizon, data in d.groupby(["horizon"]):
        comp_model = data.loc[data.model=="ensemble_comp"]
        hj_model   = data.loc[data.model=="ensembleHJ"] 

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
        
    
    #--one option
#    sns.boxplot(  x="horizon"  ,y="relative_wis",hue="model", hue_order = hue_order, data= relative_wis, boxprops=dict(alpha=.3,lw=1), fliersize=0,ax=ax,linewidth=1) 
#    sns.stripplot(x="horizon",y="relative_wis",hue="model", hue_order = hue_order, dodge=True, jitter=True,data= relative_wis,ax=ax)

#    ax.axhline(1.0,ls="--",lw=1,color="black")

    ax.tick_params(which="both",labelsize=8)

    ax.set_ylabel("WIS", fontsize=10)
    #ax.set_ylabel("Relative WIS\n(Ref. = Random Walk)", fontsize=10)
    ax.set_xlabel("",fontsize=10)

    ax.set_xticklabels(["1 week","2 week","3 week"])
    
    handles, labeles = ax.get_legend_handles_labels()
    ax.legend(handles[:2+1],["Human judgment","Computational models"],fontsize=10,ncol=1,frameon=False,loc="upper left")

#    ax.set_yticks([0,1,2,4,6,8])
#    ax.set_ylim(-0.1,9)

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

    hue_order = ["ensembleHJ","ensemble_comp"]
    
    for n,horizon in enumerate(np.arange(1,3+1)):
        ax = fig.add_subplot(gs[1,n])
        
        sns.lineplot(x="cover", y="value", hue="model", hue_order=["Human Judgment","Computational models"], data = coverages.loc[coverages.horizon==horizon] ,ax=ax, lw=1.5)

        cov = coverages.loc[coverages.horizon==horizon]

        print(cov)
        
        hj = cov.loc[cov.model=="Human Judgment"]
        ax.scatter(hj.cover,hj.value,s=10,color="blue")

        comp = cov.loc[cov.model=="Computational models"]
        ax.scatter(comp.cover,comp.value,s=10,color="red")
 
        
        handles, labeles = ax.get_legend_handles_labels()

        #if n==0:
        #    ax.legend([handles[-1]],["Random walk"],fontsize=10,ncol=1,frameon=False,loc="lower center")
        #else:
        ax.get_legend().remove()
        
        ax.plot([0.2,1],[0.2,1], color="black", ls="-", lw=1.5)
    
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

    plt.savefig("wis_and_coverage__ensembles.pdf")
    plt.savefig("wis_and_coverage__ensembles.png",dpi=350)
    
    plt.close()
