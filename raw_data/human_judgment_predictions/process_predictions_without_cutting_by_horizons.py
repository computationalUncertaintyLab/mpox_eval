#mcandrew

import sys
import numpy as np
import pandas as pd

def produce_cut_point_dataset(d,horizon):
    #--Cut point will be two weeks before the resolution time
    def subtract_weeks(x,weeks):
        import datetime
        return x - datetime.timedelta(weeks=weeks)
    d["cut_point"] = d["resolve_time"].apply(lambda x: subtract_weeks(x,weeks=horizon))

    #--submission times must be before cut point
    d_cut = d.loc[ d.time <= d.cut_point  ]

    #--most recent user submission before the cut point
    def most_recent(x):
        return  x.sort_values("time").iloc[-1]
    users_and_times  = d_cut[["user_id","question_id","time"]].drop_duplicates()
    users_and_times  = users_and_times.groupby(["user_id","question_id"]).apply(most_recent).reset_index(drop=True)

    #--merge only those users, questions, and times that are most recent
    d_cut__most_recent = d_cut.merge(users_and_times, on = ["user_id","question_id","time"]  )

    #--time horizon for these individual predictions
    d_cut__most_recent["horizon"] = horizon

    return d_cut__most_recent


def produce_cut_point_dataset_to_hours(d,horizon):
    #--Cut point will be two weeks before the resolution time
    def subtract_minutes(x,hours):
        import datetime
        return x - datetime.timedelta(hours=hours)
    d["cut_point"] = d["resolve_time"].apply(lambda x: subtract_minutes(x,hours=horizon))

    #--submission times must be before cut point
    d_cut = d.loc[ d.time <= d.cut_point  ]

    #--most recent user submission before the cut point
    def most_recent(x):
        return  x.sort_values("time").iloc[-1]
    users_and_times  = d_cut[["user_id","question_id","time"]].drop_duplicates()
    users_and_times  = users_and_times.groupby(["user_id","question_id"]).apply(most_recent).reset_index(drop=True)

    #--merge only those users, questions, and times that are most recent
    d_cut__most_recent = d_cut.merge(users_and_times, on = ["user_id","question_id","time"]  )

    #--time horizon for these individual predictions
    d_cut__most_recent["horizon"] = horizon

    return d_cut__most_recent

if __name__ == "__main__":

    predictions = pd.read_csv("./predictions.csv", sep=";")

    #--restrict to questions with a resolution
    predictions = predictions.loc[~np.isnan(predictions.resolution)]

    #--separate into continuous and binary predictions
    continuous_predictions = predictions.loc[predictions.question_type=="continuous"]
    continuous_predictions = continuous_predictions.drop(columns = ["binary_prediction"])

    #--from wide to long format
    continuous_predictions = continuous_predictions.melt( id_vars = ['question_id', 'user_id', 'time', 'void', 'question_type', 'resolution','resolve_time', 'close_time'])
    
    #--remove extra text around r value
    continuous_predictions["scaled_value"] = [float(_[0]) for _ in continuous_predictions.variable.str.findall("(\d+[.]\d+|\d+)")]

    #--save all continuous predictions submitted before the close time
    continuous_predictions.to_csv("./all_continuous_predictions.csv",index=False)

    #--format columns
    continuous_predictions["time"] = continuous_predictions.time.astype('datetime64[ns]')
    continuous_predictions["resolve_time"] = continuous_predictions.resolve_time.astype('datetime64[ns]')

    #--from scaled value to values on the original scale
    scale_params = pd.read_csv("./scaling_data.csv")
    continuous_predictions = continuous_predictions.merge(scale_params, on = ["question_id"])

    #--this density corrsponds to the [0,1] scale
    continuous_predictions = continuous_predictions.rename(columns = {"value":"scaled_density_value"})
    
    def add_original_scale(row):
        deriv_ratio = row["deriv_ratio"]
        minvalue    = row["min"]
        maxvalue    = row["max"]

        numValues = 101 #THIS IS HARDCODED
        
        if deriv_ratio==1:
            original_value = minvalue + (maxvalue - minvalue)*row.scaled_value
        else:
            exponent = np.log(deriv_ratio)
            b = (maxvalue-minvalue)/(deriv_ratio-1.)
            original_value = b* np.exp( exponent*row.scaled_value)
        return original_value
        
    continuous_predictions["original_value"] = continuous_predictions.apply(add_original_scale,1)

    #--add proper density values
    def add_original_density(row):
        deriv_ratio = row["deriv_ratio"]
        minvalue    = row["min"]
        maxvalue    = row["max"]

        numValues = 101 #THIS IS HARDCODED
        
        if deriv_ratio==1:
            dens =row["scaled_density_value"]*(1./(maxvalue-minvalue)) 
        else:
            exponent = np.log(deriv_ratio)
            b = (maxvalue-minvalue)/(deriv_ratio-1.)

            #ys*(1./(xs-a))*(1./phi)
            dens = row["scaled_density_value"]*(1./(row.original_value))*(1./exponent)
        return dens
    continuous_predictions["value"] = continuous_predictions.apply(add_original_density,1)
    
    continuous_predictions.to_csv("./continuous_no_horizons_predictions.csv",index=False)

