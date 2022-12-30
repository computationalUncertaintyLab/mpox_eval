#mcandrew

import sys
sys.path.append("../../")

from mods.index import index

import numpy as np
import pandas as pd

from datetime import timedelta

class model(object):
    def __init__(self, data):
        self.data = data
        #self.add_model_days()
        self.all_locations()
        self.horizon = 28 #--28 day horizon

    def all_locations(self):
        self.all_locations = self.data.location.unique()
        
    def add_model_days(self):
        import sys
        sys.path.append("../../")

        from mods.time_help import from_date_to_model_day
        
        self.data["model_days"] = self.data.apply(lambda x: from_date_to_model_day(x.Day),1)
        
    def stan_model(self):
        self.comp_model = '''
        data {
           int N;
           int horizon;
           int last_day;           

           vector [N] days;
           //vector [N] cases;
           int cases [N];

        }
        parameters {
            real alpha;
            real beta;
            real t;

            real<lower=0> sigma;
        }
        model {
            for (n in 1:N){
                 cases[n] ~ neg_binomial_2( exp2(alpha+ beta*(days[n] - t)), sigma );
            }
        }  
        generated quantities {
            vector[last_day + horizon] cases_new;
            for (n in 1:last_day+horizon){
               cases_new[n] = neg_binomial_2_rng( exp2(alpha+ beta*(n-t)) , sigma);
            }
        }
        '''
    def forecast_location(self,location):
        import stan
        
        loc_data = self.data.loc[self.data.location==location]
        loc_data = loc_data.loc[:,["model_days","values"]]
        self.loc_data  = loc_data

        d = loc_data.to_numpy()

        #--reference date
        ref = min(d[:,0])
        self.reference_model_day = ref
        
        #--fit model
        data = {"days": d[:,0] - ref
                ,"cases":[int(x) for x in d[:,-1]]
                ,"last_day":int(max(d[:,0]))
                ,"N": d.shape[0]
                ,"horizon":self.horizon}
        self.stan_model()

        posterior = stan.build(self.comp_model, data=data)
        self.fit = posterior.sample(num_samples=5*10**3,num_chains=1)
        return self.fit

if __name__ == "__main__":

    #--data. First column is time and second column is values to forecast

    def reformat_data(x,loc,old_name="cumulative_cases"):
        x["model_days"] = np.arange(0,len(x))
        x["location"]        = loc
        
        x = x.rename(columns= {old_name:"values"})
        
       
        return x
    
    #--QID 10979
    us_cases = pd.read_csv("../../data/usa_cases_from_globalhealth.csv")
    us_cases = reformat_data(us_cases,"us_cases")
    us_cases["question_id"] = 10979
    
    #--QID 11039
    canada_cases = pd.read_csv("../../data/canada_cases_from_globalhealth.csv")
    canada_cases = reformat_data(canada_cases,"canada_cases")
    canada_cases["question_id"] = 11039

    #--QID 10981
    us_states = pd.read_csv("../../data/number_of_us_states.csv")
    us_states = reformat_data(us_states,"us_states","states")
    us_states["question_id"] = 10981

    us_states = us_states.rename(columns = {"time_stamp":"Date_entry"})
    
    #--QID 10978
    europe_cases = pd.read_csv("../../data/europe_cases_from_globalhealth.csv")
    europe_cases = reformat_data(europe_cases,"europe_cases")
    europe_cases["question_id"] = 10978
        
    #--QID 10975:
    all_countries = pd.read_csv("../../data/countries_from_globalhealth.csv")
    all_countries = reformat_data(all_countries,"all_countries","counts")
    all_countries["question_id"] = 10975

    training_data = us_cases.append( canada_cases ).append(us_states).append(europe_cases).append(all_countries) 

    #--cutpoint data
    cutpoint = pd.read_csv("../../data/cut_points_resolve_times_qids.csv")

    training_data = training_data.merge(cutpoint, on = ["question_id"] )

    #--collect specific columns and tranform columns to date time objects
    training_data = training_data[["model_days","location","question_id","values","cut_point","resolve_time","Date_entry","horizon"]]
    training_data["Date_entry"] = pd.to_datetime(training_data["Date_entry"])
    training_data["cut_point"] = pd.to_datetime(training_data["cut_point"])

    #--run through all training data and build a model
    all_samples = pd.DataFrame()
    
    for (loc,qid,hor,cut,res), subset in training_data.groupby(["location","question_id","horizon","cut_point","resolve_time"]):
        data = subset.loc[subset.Date_entry<=subset.cut_point]

        times = [pd.Timestamp(x) for x in data.Date_entry.values]

        #--add more time units
        last_time      = times[-1]
        last_model_day = int(data.model_days.values[-1] )

        last_model_day__orig = last_model_day
        last_truth_day = subset.iloc[-1]["model_days"]
        
        time_data = { "times":times, "model_days": list(data.model_days.values)}
        for _ in range(27):
            last_time      =  pd.Timestamp(last_time) + timedelta(days=1)
            last_model_day +=1
            
            time_data["times"].append(last_time) 
            time_data["model_days"].append(last_model_day)
        time_data = pd.DataFrame(time_data)
            
        data = data[["model_days","location","values"]]
        
        mdl = model( data )
        mdl.forecast_location(loc)

        samples = mdl.fit.get("cases_new")
        nrow,ncol = samples.shape

        samples_fmt = {"value":[], "sample":[], "sim_num":[], "model_days":[]}
        for ahead,sample in enumerate(samples):
            samples_fmt["value"].extend(sample)

            samples = np.arange(0,ncol)
            samples_fmt["sample"].extend(samples)
            
            samples_fmt["model_days"].extend( samples + 1 + last_model_day__orig  )
            samples_fmt["sim_num"].extend([ahead]*ncol)
            
        samples_fmt = pd.DataFrame(samples_fmt)

        samples_fmt["horizon"]      = hor
        samples_fmt["cut_point"]    = cut
        samples_fmt["resolve_time"] = res
        samples_fmt["target"]       = loc
        samples_fmt["question_id"]  = qid
        
        samples_fmt = samples_fmt.merge(time_data, on = ["model_days"])

        #--cut to the evaluation time point
        samples_fmt = samples_fmt.loc[samples_fmt.model_days==last_truth_day]
 

        all_samples = all_samples.append(samples_fmt)

    all_samples["model"] = "WAG"
    all_samples.to_csv("all_samples.csv",index=False)

    #--construct quantiles from the samples
    def build_quantiles(d):
        quantiles = list(100*np.arange(0.01,0.99+0.01,0.01))
        quantiles_values = np.percentile( d.value, quantiles )

        return pd.DataFrame({"quantile":quantiles, "value":quantiles_values})
    all_quantiles = all_samples.groupby(["target","question_id","horizon","cut_point","resolve_time"]).apply(build_quantiles)
    all_quantiles = all_quantiles.reset_index()
    
    all_quantiles["model"] = "WAG"
    all_quantiles.to_csv("all_quantiles.csv",index=False)
