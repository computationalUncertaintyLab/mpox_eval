#mcandrew

import sys
import numpy as np
import pandas as pd

from datetime import datetime

if __name__ == "__main__":
    cutpoints_and_truth = pd.read_csv("./cut_points_resolve_times_qids.csv")

    truth = {"question_id":[], "horizon":[], "truth":[], "date":[]}

    #--QID 11039
    canada = pd.read_csv("../raw_data/observed_data/canada_cases_from_globalhealth.csv")
    canada["date"] = pd.to_datetime(canada["Date_entry"])

    for horizon, data in cutpoints_and_truth.loc[cutpoints_and_truth.question_id==11039].groupby("horizon"):

        truth_date = datetime.strptime(data.cut_point.values[0],"%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        true_value = canada.loc[canada["date"] == truth_date, "cumulative_cases"]

        truth["question_id"].append(11039)
        truth["horizon"].append(horizon)
        truth["date"].append(truth_date)
        truth["truth"].append(float(true_value))

    #--QID 10981
    number_of_us_states = pd.read_csv("../raw_data/observed_data/number_of_us_states.csv")
    number_of_us_states["time_stamp"] = pd.to_datetime(number_of_us_states["time_stamp"])
    
    for horizon, data in cutpoints_and_truth.loc[cutpoints_and_truth.question_id==10981].groupby("horizon"):

        truth_date = datetime.strptime(data.cut_point.values[0],"%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        true_value = number_of_us_states.loc[number_of_us_states["time_stamp"] == truth_date, "states"]

        truth["question_id"].append(10981)
        truth["horizon"].append(horizon)
        truth["date"].append(truth_date)

        try:
            truth["truth"].append(float(true_value))
        except TypeError:
            truth["truth"].append(-1.)
            
    #--QID 10979
    usa = pd.read_csv("../raw_data/observed_data/usa_cases_from_globalhealth.csv")
    usa["date"] = pd.to_datetime(usa["Date_entry"])

    for horizon, data in cutpoints_and_truth.loc[cutpoints_and_truth.question_id==10979].groupby("horizon"):

        truth_date = datetime.strptime(data.cut_point.values[0],"%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        true_value = usa.loc[usa["date"] == truth_date, "cumulative_cases"]

        truth["question_id"].append(10979)
        truth["horizon"].append(horizon)
        truth["date"].append(truth_date)
        truth["truth"].append(float(true_value))

    #--QID 10978
    europe = pd.read_csv("../raw_data/observed_data/europe_cases_from_globalhealth.csv")
    europe["date"] = pd.to_datetime(europe["Date_entry"])

    for horizon, data in cutpoints_and_truth.loc[cutpoints_and_truth.question_id==10978].groupby("horizon"):

        truth_date = datetime.strptime(data.cut_point.values[0],"%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        true_value = europe.loc[europe["date"] == truth_date, "cumulative_cases"]

        truth["question_id"].append(10978)
        truth["horizon"].append(horizon)
        truth["date"].append(truth_date)
        truth["truth"].append(float(true_value))
    
    #--QID 10975
    europe = pd.read_csv("../raw_data/observed_data/countries_from_globalhealth.csv")
    europe["date"] = pd.to_datetime(europe["Date_entry"])

    for horizon, data in cutpoints_and_truth.loc[cutpoints_and_truth.question_id==10975].groupby("horizon"):

        truth_date = datetime.strptime(data.cut_point.values[0],"%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        true_value = europe.loc[europe["date"] == truth_date, "counts"]

        truth["question_id"].append(10975)
        truth["horizon"].append(horizon)
        truth["date"].append(truth_date)
        truth["truth"].append(float(true_value))
   
    truth = pd.DataFrame(truth)

    truth.to_csv("./truths_at_each_horizon.csv")
