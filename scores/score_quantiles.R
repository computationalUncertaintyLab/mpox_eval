library(scoringutils)
library(magrittr)
library(dplyr)

#--import forecasts
ensemble_quantiles__HJ = read.csv("../models/equally_weighted_hj_ensemble/all_quantiles.csv")
ensemble_quantiles__HJ$model="HJE"

wag_model_quantiles     = read.csv("../models/wag/all_quantiles_numpyro.csv")
ll_model_quantiles      = read.csv("../models/latent_local/all_quantiles_numpyro.csv") 
rw_model_quantiles      = read.csv("../models/random_walk/all_quantiles_numpyro.csv")
comp_ensemble_quantiles = read.csv("../models/computational_ensemble/all_quantiles_numpyro.csv")

#--convert percentiles to quantiles
wag_model_quantiles$quantile = wag_model_quantiles$quantile/100
ll_model_quantiles$quantile = ll_model_quantiles$quantile/100
rw_model_quantiles$quantile = rw_model_quantiles$quantile/100
comp_ensemble_quantiles$quantile = comp_ensemble_quantiles$quantile/100

#--collect columns needed
needed_cols = c("question_id","model","horizon","quantile","value")
ensemble_quantiles__HJ  = ensemble_quantiles__HJ[,needed_cols]
wag_model_quantiles     =  wag_model_quantiles[,needed_cols]
ll_model_quantiles      = ll_model_quantiles[,needed_cols]
rw_model_quantiles      = rw_model_quantiles[,needed_cols]
comp_ensemble_quantiles = comp_ensemble_quantiles[,needed_cols]


#--stack all forecasts
compute_wis_and_mae = function(x){
  #--merge in the truth
  truth = read.csv("../truth_tables/truths.csv")
  x = merge( x, truth, by = c("question_id") )
  
  x = rename( x, observed = truth, predicted=value, quantile_level=quantile)
  
  x = x[, c("observed","quantile_level","predicted","model","horizon","question_id") ]
  
  forecasts = as_forecast(x)
  
  scores <- forecasts |> 
    score()
  
  #--dont need all these columns
  scores = scores[,c("model","horizon","question_id","wis","ae_median")]
  scores = rename(scores, WIS=wis, MAE=ae_median)
  
  return(scores)
}

HJE_scores = compute_wis_and_mae(ensemble_quantiles__HJ)
WAG_scores = compute_wis_and_mae(wag_model_quantiles)
LL_scores  = compute_wis_and_mae(ll_model_quantiles)
RW_scores  = compute_wis_and_mae(rw_model_quantiles)
CE_scores  = compute_wis_and_mae(comp_ensemble_quantiles)

scores     = rbind(HJE_scores, WAG_scores)
scores     = rbind(scores, LL_scores)
scores     = rbind(scores, RW_scores)
scores     = rbind(scores, CE_scores)

write.csv(scores,"WIS_scoresv2.csv")


