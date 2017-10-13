
# 

# eda_p_t

#%%
from __future__ import print_function

import numpy as np               #%matplotlib qt
from pandas import read_csv
from pandas import datetime

import matplotlib.pyplot as plt    #from matplotlib import pyplot

from pandas.plotting import autocorrelation_plot as atp   # from pandas.tools.plotting import autocorrelation_plot

import scipy.stats as stats
import statsmodels.api as stm

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

from pandas import DataFrame
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

from pandas.tools.plotting import lag_plot
from pandas import TimeGrouper

#%%
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
#%%    
# read SPY data (SPY is a S&P500 ETF, which is a proxy of the S&P500 cash index)    
s_tn = read_csv('spy_dt_wk_1993_2017.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)  #, date_parser=parser
p_t=s_tn['Close']     # just get the closing prices
print(s_tn.head())

#%%
# plot the price time series p_t
fig1 = plt.figure()
#ax1 = fig1.add_subplot(111)
#ax1.plot(p_t)

p_t.plot()
#plt.show()

#%%
# calculate return series and plot it
pr_t=p_t.pct_change(1)

#%%
fig1p1 = plt.figure()
pr_t.abs().plot()

plt.title('Weekly absolute returns of SPY for years 1993-2017')

#%%
# plot autocorrelation function of p_t s_tn
fig2 = plt.figure()    # ax2 = fig2.add_subplot(111)
atp(p_t)               #plt.show()

#%%
# plot a lag plot of the series p_t
fig3 = plt.figure()
lag_plot(pr_t)       # this is a lag plot with 1-step lag

#%%
# plot a sample histogram
fig4 = plt.figure()
pr_t.hist(bins=50)        # plt.hist(pr_t,50)      #

#sp_t=p_t[:,1]
#%%  
# plot the probability density estimate
fig5 = plt.figure()
pr_t.plot(kind='kde')    

#%%
# plot autocorrelation function of the series p_t
#fig6 = plt.figure()
plot_acf(p_t, lags=500)     # 

#atp(T_t) 

#%%
# plot partial autocorrelation function of the series p_t
#fig7 = plt.figure()
plot_pacf(pr_t, lags=50)

#%%
# make a qq-plot against a standard distribution of some type
fig7 = plt.figure()
stm.qqplot(pr_t, line='q')    # , stats.t, distargs=(5,) #, line='45' 

plt.title('QQ-plot of SPY ETF weekly returns for years 1993-2017')
plt.xlabel('Quantiles of standard normal distribution')


#%%
# group the price series by years to compare performance between different years
gp_y = p_t.groupby(TimeGrouper('A'))    # '_y'=by-year, 'gp'=group-by-object of price
Nw=gp_y.size().max()                    # max nr of weeks across all years in gp_y

p_ty = DataFrame(index=range(0,Nw))   # i need to have same _t sizes for all years in _y   #.reindex_like()
for name, group in gp_y:
   p_w = group.values                 # '_w'=week  # p_w.size
   
   if Nw > p_w.size :                         # need to have same sizes for all years
      dp = np.full((1,Nw - p_w.size),np.nan)
      p_w = np.append(p_w,dp)                 # pad p_w until we get it to size Nw 
   # end if
   
   p_w = (p_w - p_w[0])/p_w[0]     #p_w - p_w[0]  # subtract first element to compare different years      
   p_ty[name.year] = p_w           #print(name)  #group.size
# end for name, group
   
#%%
# plot time series heatmap   
plt.matshow(p_ty, interpolation=None, aspect='auto')   #
plt.show()





