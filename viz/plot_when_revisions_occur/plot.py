#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime,timedelta

from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.gridspec as gridspec

def mm2inch(x):
    return x/25.4

if __name__ == "__main__":

    wis = pd.read_csv("../../scores/revisions/WIS_scores.csv")
    pit = pd.read_csv("../../scores/revisions/PITS.csv")

    truths = pd.read_csv("../../truth_tables/truths.csv")

    wis = wis.merge(truths, on = ["question_id"])
    pit = pit.merge(truths, on = ["question_id"])

    creation_times = pd.read_csv("../../raw_data/human_judgment_predictions/questions_and_creattion_time.csv")

    wis = wis.merge(creation_times, on = ["question_id"])
    pit = pit.merge(creation_times, on = ["question_id"])

           
    #--add revision number to these datasets
    def add_revision_number(x,y):
    
        x = x.sort_values(y)
        x["revision_number"] = np.arange(0,len(x))

        truth_time = datetime.strptime(x.iloc[0]["date"],"%Y-%m-%d")

        revision_times = []
        for _,row in x.iterrows():
            delta = datetime.strptime(row[y],"%Y-%m-%d %H:%M:%S") - truth_time
            revision_times.append( abs(delta.total_seconds()/(60*60*24))  )
        x["time_from_revision_to_truth"] = revision_times
        
        return x
    
    wis = wis.groupby(["question_id","user_id"]).apply(lambda x: add_revision_number(x,"horizon")).reset_index()
    pit = pit.groupby(["question_id","user_id"]).apply(lambda x: add_revision_number(x,"time")).reset_index()

    #--erroneously high WIS score
    wis = wis.loc[wis.WIS<47000]


    #--time from open to close
    wis["open_to_close"] = [ (datetime.strptime(x,"%Y-%m-%d") - datetime.strptime(y,"%Y-%m-%d")).total_seconds()/(60*60*24) for x,y in zip(wis.date,wis.open_time) ]

    #--
    fromQid2Labels = { 11039:"Ttl cases in Canada"
                      ,10981:"Num of US states"
                      ,10979:"Ttl cases in US"
                      ,10978:"Ttl cases in Europe"
                      ,10975:"Num of Countries"
                      ,10977:"PHEIC"}

    
    #--how long until someone produces a forecast
    first_submissions = wis.loc[wis.revision_number==0]
    
    def number_of_hours(x):
        opentime = datetime.strptime(x["open_time"],"%Y-%m-%d") 
        delta = datetime.strptime(x["horizon"],"%Y-%m-%d %H:%M:%S")
        diff = delta - opentime
        return abs(diff.total_seconds()/(60*60*24))
    first_submissions["open_to_submit"] = first_submissions.apply(number_of_hours,1)
    
    def cdf(x):
        x,n = np.sort(x.open_to_submit.values),len(x)
        d = pd.DataFrame({"x":x,"px":np.arange(1.,n+1)/n})
        return d
    cdfs = first_submissions.groupby(["question_id"]).apply(cdf).reset_index()
    
    def estimate_percs(x):
        from scipy.interpolate import interp1d
        from scipy.integrate import quad
        from scipy.optimize import root_scalar

        xs = x.x.values
        ys = x.px.values
        
        F = interp1d(xs,ys)

        q2v = {"q":[],"v":[]}
        for q in [0.25,0.50,0.75]:
            i = np.argmin(abs(ys-q))
            if ys[i]<q:
                l,u = xs[i], xs[i+1]
            else:
                l,u = xs[i-1], xs[i]
            g = lambda x: F(x)-q
            q2v["q"].append(q)

            result = root_scalar(g, bracket = (l,u))

            q2v["v"].append(result.root)

        return pd.DataFrame(q2v)
    quantiles = cdfs.groupby(["question_id"]).apply(estimate_percs).reset_index()

    plt.style.use("fivethirtyeight")

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2
                          ,left=0.10
                          ,right=0.95
                          ,bottom=0.10
                          ,top=0.95
                          ,wspace=0.10
                          ,hspace=0.20)
    
   # fig,axs = plt.subplots(4,2, height_ratios=[4,1]*2 )

    colors = sns.color_palette("colorblind",6)

    sub_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,0],height_ratios=[3.5,1]*1, hspace=0.025 )
    
    ax0 =  fig.add_subplot(sub_gs[0])
    ax1 =  fig.add_subplot(sub_gs[1])
    
    #--plotting
    for n,(qid,x) in enumerate(quantiles.groupby(["question_id"])):

        subset = cdfs.loc[cdfs.question_id==qid]
        ax = ax0
        ax.step( subset.x.values, subset.px.values, color = colors[n], lw=2,alpha=0.75  )

        ax.set_xlim(0,75)
        ax.set_ylim(-0.025,1.025)

        ax.set_yticks(np.arange(0,1+0.25,0.25))
        ax.set_xticks( list(np.arange(0,28+14,7*2)) + [40,43,73] )

        ax.set_xticklabels([])
        
        #--quantiles
        ax = ax1
        x["q"] = np.round(x.q,2)
        x = x.set_index(["q"])
        
        ax.plot( [ x.loc[0.25]["v"], x.loc[0.75]["v"]],[n]*2, lw=5, color = colors[n], alpha=0.5)
        ax.scatter( x.loc[0.50]["v"], n, s=10 )

        #--closetime
        close = wis.loc[wis.question_id==qid].iloc[0]["open_to_close"]
        ax.scatter( [close], [n], s=10, color="black" )

        ax.set_xlim(0,75)

        ax.set_yticks(np.arange(0,n+1))
        ax.set_yticklabels([])
        
        ax.set_xticks( list(np.arange(0,28+14,7*2)) + [40,43,73] )
        
    ax0.set_ylabel("Cumulative density",fontsize=10)    
    ax1.set_xlabel("Time until first forecast (days)", fontsize=10)
    ax1.set_ylim([-0.5,4.5])
    ax1.get_xticklabels()[4].set_horizontalalignment("left")

    #--stamp
    ax0.text(0.01,0.99,s="A.",fontsize=10,fontweight="bold",ha="left",va="top",transform=ax0.transAxes)
    

    #--when do the most revisions occur?

    submissions = wis[["question_id","user_id","horizon","revision_number","time_from_revision_to_truth"]]
   
    def cdf(x):
        x,n = np.sort(x.time_from_revision_to_truth.values),len(x)
        d = pd.DataFrame({"x":x,"px":np.arange(1.,n+1)/n})
        return d
    cdfs = submissions.groupby(["question_id"]).apply(cdf).reset_index()
    
    def estimate_percs(x):
        from scipy.interpolate import interp1d
        from scipy.integrate import quad
        from scipy.optimize import root_scalar

        xs = x.x.values
        ys = x.px.values
        
        F = interp1d(xs,ys)

        q2v = {"q":[],"v":[]}
        for q in [0.25,0.50,0.75]:
            i = np.argmin(abs(ys-q))
            if ys[i]<q:
                l,u = xs[i], xs[i+1]
            else:
                l,u = xs[i-1], xs[i]
            g = lambda x: F(x)-q
            q2v["q"].append(q)

            result = root_scalar(g, bracket = (l,u))

            q2v["v"].append(result.root)

        return pd.DataFrame(q2v)
    quantiles = cdfs.groupby(["question_id"]).apply(estimate_percs).reset_index()

    #--plotting
    sub_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,1],height_ratios=[3.5,1]*1,hspace=0.025 )
    
    ax0 =  fig.add_subplot(sub_gs[0])
    ax1 =  fig.add_subplot(sub_gs[1])
    
    for n,(qid,x) in enumerate(quantiles.groupby(["question_id"])):

        subset = cdfs.loc[cdfs.question_id==qid]
        ax = ax0
        ax.step( subset.x.values, subset.px.values, color = colors[n], lw=2,alpha=0.75, label = fromQid2Labels[qid]  )

        ax.set_xlim(0,75)
        ax.set_ylim(-0.025,1.025)

        ax.set_yticks(np.arange(0,1+0.25,0.25))
        ax.set_xticks( list(np.arange(0,28+14,7*2)) + [40,43,73] )
        
        ax.set_xticklabels([])
        
        #--quantiles
        ax = ax1
        x["q"] = np.round(x.q,2)
        x = x.set_index(["q"])
        
        ax.plot( [ x.loc[0.25]["v"], x.loc[0.75]["v"]],[n]*2, lw=5, color = colors[n], alpha=0.5)
        ax.scatter( x.loc[0.50]["v"], n, s=10 )

        close = wis.loc[wis.question_id==qid].iloc[0]["open_to_close"]
        ax.scatter( [close], [n], s=10, color="black" )

        ax.set_xlim(0,75)

        ax.set_yticks(np.arange(0,n+1))
        ax.set_yticklabels([])
        
        ax.set_xticks( list(np.arange(0,28+14,7*2)) + [40,43,73] )
    
    ax1.set_xlabel("Time of submitted forecast (days)", fontsize=10)
    ax1.set_ylim([-0.5,4.5])
    ax1.get_xticklabels()[4].set_horizontalalignment("left")

    #--stamp
    ax0.text(0.01,0.99,s="B.",fontsize=10,fontweight="bold",ha="left",va="top",transform=ax0.transAxes)
 

    #--final revision
    def last_submit(x):
        max_rev = max(x.revision_number)
        return x.loc[x.revision_number==max_rev]

    last_submissions = wis.groupby(["question_id","user_id"]).apply(last_submit).reset_index(drop=True)
    last_submissions["open_to_submit"] = last_submissions.apply(number_of_hours,1)
    
    def cdf(x):
        x,n = np.sort(x.time_from_revision_to_truth.values),len(x)
        d = pd.DataFrame({"x":x,"px":np.arange(1.,n+1)/n})
        return d
    cdfs = first_submissions.groupby(["question_id"]).apply(cdf).reset_index()
    
    def estimate_percs(x):
        from scipy.interpolate import interp1d
        from scipy.integrate import quad
        from scipy.optimize import root_scalar

        xs = x.x.values
        ys = x.px.values
        
        F = interp1d(xs,ys)

        q2v = {"q":[],"v":[]}
        for q in [0.25,0.50,0.75]:
            i = np.argmin(abs(ys-q))
            if ys[i]<q:
                l,u = xs[i], xs[i+1]
            else:
                l,u = xs[i-1], xs[i]
            g = lambda x: F(x)-q
            q2v["q"].append(q)

            result = root_scalar(g, bracket = (l,u))

            q2v["v"].append(result.root)

        return pd.DataFrame(q2v)
    quantiles = cdfs.groupby(["question_id"]).apply(estimate_percs).reset_index()

    #--plotting
    sub_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1,0],height_ratios=[3.5,1]*1,hspace=0.025 )
    
    ax0 =  fig.add_subplot(sub_gs[0])
    ax1 =  fig.add_subplot(sub_gs[1])
 
    for n,(qid,x) in enumerate(quantiles.groupby(["question_id"])):

        subset = cdfs.loc[cdfs.question_id==qid]
        ax = ax0
        ax.step( subset.x.values, subset.px.values, color = colors[n], lw=2,alpha=0.75  )

        ax.set_xlim(0,75)
        ax.set_ylim(-0.025,1.025)

        ax.set_yticks(np.arange(0,1+0.25,0.25))
        ax.set_xticks( list(np.arange(0,28+14,7*2)) + [40,43,73] )

        ax.set_xticklabels([])
        
        #--quantiles
        ax = ax1
        x["q"] = np.round(x.q,2)
        x = x.set_index(["q"])
        
        ax.plot( [ x.loc[0.25]["v"], x.loc[0.75]["v"]],[n]*2, lw=5, color = colors[n], alpha=0.5)
        ax.scatter( x.loc[0.50]["v"], n, s=10 )

        close = wis.loc[wis.question_id==qid].iloc[0]["open_to_close"]
        ax.scatter( [close], [n], s=10, color="black" )
        
        ax.set_xlim(0,75)

        ax.set_yticks(np.arange(0,n+1))
        ax.set_yticklabels([])
        
        ax.set_xticks( list(np.arange(0,28+14,7*2)) + [40,43,73] )
    
    ax0.set_ylabel("Cumulative density",fontsize=10)    
    ax1.set_xlabel("Time from last forecast to close (days)", fontsize=10)
    ax1.set_ylim([-0.5,4.5])
    ax1.get_xticklabels()[4].set_horizontalalignment("left")

    #--stamp
    ax0.text(0.01,0.99,s="C.",fontsize=10,fontweight="bold",ha="left",va="top",transform=ax0.transAxes)

    #--wis score correlate with time to revision
    ax =  fig.add_subplot(gs[1,1])
    def norm_wis(x):
        wis = x["WIS"]
        m,s = np.mean(wis), np.std(wis)

        x["nwis"] = (wis-m)/s

        return x
    wis = wis.groupby(["question_id"]).apply(norm_wis).reset_index()


    for n,(qid, subset) in enumerate(wis.groupby(["question_id"])):
        xs,ys = subset.time_from_revision_to_truth, subset.nwis
        
        ax.scatter(xs,ys, s=10
                   #, color=colors[n]
                   , facecolors='none'
                   , edgecolors=colors[n]
                   ,lw=1
                   , alpha=0.3  )
        b1,b0 = np.polyfit(xs,ys,1)

        model_x_and_modely = lowess(ys,xs)
        modelx, modely = model_x_and_modely[:,0], model_x_and_modely[:,1]
        
        #minx,maxx = min(xs), max(xs)
        #ax.plot([minx,maxx],[b0+b1*minx, b0+b1*maxx], color = colors[n], lw=1,alpha=0.80)
        ax.plot(modelx,modely, color = colors[n], lw=1,alpha=1., label = fromQid2Labels[qid])

    ax.invert_xaxis()
    
    ax.legend(frameon=False,fontsize=9,loc="upper left",ncol=2,labelspacing=0.25, columnspacing=0.5, handlelength=1.0, bbox_to_anchor=(0.01,0.975 ))
    
    ax.set_ylabel("Normalized WIS", fontsize=10)
    ax.set_xlabel("Time from forecast to close (days)", fontsize=10)

    #--stamp
    ax.text(0.01,0.99,s="D.",fontsize=10,fontweight="bold",ha="left",va="top",transform=ax.transAxes)


    
    #--all plots
    for ax in fig.get_axes():
        ax.tick_params(which="both",labelsize=8)
        
    w = mm2inch(183)
    fig.set_size_inches(w,w/1.5)

    fig.set_tight_layout(True)

    plt.savefig("revisions.pdf")
    plt.savefig("revisions.png", dpi=350)
    
    plt.close()




    
    # #--number of revisions
    # def final_revisision(x):
    #     return pd.Series({"final_rev":x.revision_number.max()})
    # revisions = submissions.groupby(["question_id","user_id"]).apply(final_revisision).reset_index()

    # def ccdf(x):
    #     x,n = np.sort(x.final_rev.values),len(x)
    #     d = pd.DataFrame({"x": np.log10(x+1),"px":np.log10(1.-np.arange(1.,n+1)/n)})
    #     return d
    # ccdfs = revisions.groupby(["question_id"]).apply(ccdf).reset_index()
    

    # #--removve 10975
    # #wis = wis.loc[wis.question_id!=10975]

    # #--change direction of time to revision
    # wis["neg_time"] = -1*wis.time_from_revision_to_truth
    
    # #--regress on WIS
    # def regress(x):
    #     print(x)
        
    #     mod  = smf.ols("WIS ~ revision_number+neg_time", x)
    #     modf = mod.fit()
    #     epsilon__wis = modf.resid_pearson

    #     #--regress on revision_number
    #     mod  = smf.ols("revision_number~neg_time", x)
    #     modf = mod.fit()
    #     epsilon__revisions = modf.resid_pearson

    #     plt.scatter(epsilon__revisions, epsilon__wis, s=10, alpha=0.5)
    #     plt.show()

    # wis.groupby(["question_id"]).apply(regress)
    
    # wis["eps_wis"] = epsilon__wis
    # wis["eps_rev"] = epsilon__revisions

    
    # plt.plot( epsilon__wis,epsilon__revisions, "ko", alpha=0.5 ) ;plt.show()

    
    # plt.show()
