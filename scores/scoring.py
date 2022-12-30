#mcandrew

def IS(quantiles,y,alpha):
    l,u = quantiles
    return u-l + (2/alpha)*(l-y)*(1 if y <l else 0) + (2/alpha)*(y-u)*(1 if y >u else 0)

def WIS(quantiles,y=None):
    import pandas as pd   
    '''
    quantiles: A list of tuples that describes each of the (1-alpha) central predictions intervals. 
    A (1-alpha) central prediction intervals is a pair of values such that we expect the truth to fall 
    between those two numebrs with probability (1-alpha). For example, a (1-0.05) central prediction 
    interval---[l,u]---is an interval such that we assign a probability of 0.95 to the truth falling 
    inside of [l,u].
    '''

    #--collect truth from quantiles file
    if y is None:
        y = float(quantiles.iloc[0]["truth"])
    
    #--find median value
    median = quantiles.loc[ quantiles["quantile"]==0.50,"value" ].values

    #--collect all lower quantiles and determine the alphas
    lower_quantiles = list(quantiles.loc[ quantiles["quantile"] < 0.50, "quantile" ])
    alphas = [ round(1 - 2*q,2) for q in lower_quantiles]

    #--compute WIS
    K = len(alphas)
    WIS = (1/2)*abs(y-median)
    for alpha in alphas:

        lower_alpha = round((1-alpha)/2,3)
        upper_alpha = round( 1+(alpha-1)/2,3)
    
        lower = float(quantiles.loc[ quantiles["quantile"] == lower_alpha,"value" ].values)
        upper = float(quantiles.loc[ quantiles["quantile"] == upper_alpha,"value"].values)
        
        WIS+= (alpha/2)*IS([lower,upper],y,alpha)
    WIS*=1/(K+0.5)
    return float(WIS)

if __name__ == "__main__":
    pass
    

