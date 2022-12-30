#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime,timedelta

if __name__ == "__main__":


    wis = pd.read_csv("../../scores/revisions/WIS_scores.csv")
    pit = pd.read_csv("../../scores/revisions/PITS.csv")

    truths = pd.read_csv("../../truth_tables/truths.csv")

    wis = wis.merge(truths, on = ["question_id"])
    pit = pit.merge(truths, on = ["question_id"])

            
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
        
    sns.lineplot( x="time_from_revision_to_truth", y="PIT", style="question_id",hue="user_id", data = pit )
    plt.show()
