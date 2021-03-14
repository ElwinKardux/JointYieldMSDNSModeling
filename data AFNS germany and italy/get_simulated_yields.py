import os as os
import sys as sys 
path = os.path.dirname(os.path.realpath('__file__'))
from data_statistics import data_statistics
import numpy as np
import pandas as pd
from datetime import time
import pandas_market_calendars as mcal
from statsmodels.tsa import stattools
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set(palette = 'bright',color_codes=True)
from scipy import stats
import string

maturities_in_years = np.array([1,2,5,10,20,30])

# read in monthly historical nominal yields
denom_file = 'data AFNS germany and italy/denom interpolated.csv' #!
denom_df = pd.read_csv(os.path.join(path, denom_file), sep=',')
denom_df.set_index(pd.date_range(start='2000-01-01', end='2020-12-31',freq='BM'), inplace=True)
denom_df = denom_df[['1','2','5','10','20','30']]
denom_df.columns = maturities_in_years

# read in monthly historical real yields
deinf_file = 'data AFNS germany and italy/deinf interpolated.csv' #!
deinf_df = pd.read_csv(os.path.join(path, deinf_file), sep=',')
deinf_df.set_index(pd.date_range(start='2000-01-01', end='2020-12-31',freq='BM'), inplace=True)
deinf_df = deinf_df[['1','2','5','10','20','30']]
deinf_df.columns = maturities_in_years

# read in monthly simulated nominal and real yields
yields_file = 'data AFNS germany and italy/MSSigma_de_1.csv' #!
yields_df = pd.read_csv(os.path.join(path, yields_file), sep=',')
yields_df.set_index(pd.date_range(start='2021-01-01', end='2086-01-01',freq='BM'), inplace=True)

# create dataframes Germany
nom_yields_df = yields_df[['V1','V3','V5','V7','V9','V11']]
nom_yields_df.columns = maturities_in_years
real_yields_df = yields_df[['V2','V4','V6','V8','V10','V12']]
real_yields_df.columns = maturities_in_years

# combine back- and forwardtest: 10 (historical) + 55 years
comb_nom_yields_df = pd.concat([denom_df.loc['2010-01-29':], nom_yields_df[:'2074-12-31']])
comb_real_yields_df = pd.concat([deinf_df.loc['2010-01-29':], real_yields_df[:'2074-12-31']])

# investment of 1 and reinvestments each month (!)
nom_cum_ret = (comb_nom_yields_df/(100*12)+1).expanding().apply(np.prod,raw=True)
real_cum_ret = (comb_real_yields_df/(100*12)+1).expanding().apply(np.prod,raw=True)

# get nom_cum_ret and nom_ret for regime_hmm_train_inflation
nom_ret = nom_cum_ret/nom_cum_ret.shift(1) - 1

# plot data
comb_nom_yields_df.plot()
plt.legend(loc='upper right')
plt.xlabel("Date")
plt.ylabel("Yield in percentage")
# plt.axvline(x='2020-12-31', color= 'black', ls='--')
plt.axvspan(xmin='2010-01-29', xmax='2020-12-31', alpha=0.25)
# plt.title("Simulated nominal yields for different maturities and high inflation")

comb_real_yields_df.plot()
plt.legend(loc='upper right')
plt.xlabel("Date")
plt.ylabel("Yield in percentage")
# plt.axvline(x='2020-12-31', color= 'black', ls='--')
plt.axvspan(xmin='2010-01-29', xmax='2020-12-31', alpha=0.25)
# plt.title("Simulated real yields for different maturities and high inflation")

# save data (in special form for getting funds data)
ones_df = pd.DataFrame([pd.Series(np.ones(6))])
ones_df.columns = maturities_in_years
nom_cum_ret = pd.concat([ones_df,nom_cum_ret])
real_cum_ret = pd.concat([ones_df,real_cum_ret])

nom_cum_ret.to_csv(path  + '/data back- and forward test/denom_mss_1' + '.csv', index=False) #!
real_cum_ret.to_csv(path  + '/data back- and forward test/deinf_mss_1' + '.csv', index=False) #!


#%%
# https://aaaquants.com/2018/01/04/plotting-volatility-surface-for-options/
from pandas_datareader.data import Options
from dateutil.parser import parse
import datetime, random
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
#from implied_vol import BlackScholes
from functools import partial
from scipy import optimize
import numpy as np
from scipy.interpolate import griddata
import matplotlib.ticker as ticker

def plot3D(X,Y,Z,fig,ax,title):
   ax.plot(X,Y,Z,'o', color = 'red')
   # ax.set_title(title)
   plt.xlabel('Maturity (Years)')
   plt.ylabel('Date')
   ax.set_zlabel('Yield in percentage')
   
def mesh_plot2(X,Y,Z,fig,ax,title):
   XX,YY,ZZ = make_surf(X,Y,Z)
   # ax.set_title(title)
   ax.plot_surface(XX,YY,ZZ,cmap=cm.seismic)
   
   # ax.w_yaxis.set_major_locator(ticker.FixedLocator(some_dates)) # I want all the dates on my xaxis
   ax.w_yaxis.set_major_formatter(ticker.FuncFormatter(format_date))
   for tl in ax.w_yaxis.get_ticklabels(): # re-create what autofmt_xdate but with w_xaxis
       tl.set_ha('right')
       tl.set_rotation(30)

   ax.contour(XX,YY,ZZ)
   plt.xlabel('Maturity (Years)')
   # plt.ylabel('Date')
   ax.set_zlabel('Yield in percentage')
       
def make_surf(X,Y,Z):
   XX,YY = np.meshgrid(np.linspace(min(X),max(X),100),np.linspace(min(Y),max(Y),100))
   ZZ = griddata(np.array([X,Y]).T,np.array(Z),(XX,YY), method='linear')
   return XX,YY,ZZ

# prepare data for 3D plot
df = pd.DataFrame()
for i in np.array([1,2,5,10,20,30]):
    df1 = pd.DataFrame(comb_nom_yields_df[i]) #!
    df1.columns = ['yield']
    df1['maturity'] = i
    
    df = pd.concat([df,df1],axis=0)

# # index to int
# def to_integer(dt_time):
#     return 10000*dt_time.year + 100*dt_time.month + dt_time.day
#     # return dt_time.year

# df.index = to_integer(df.index)

def format_date(x, pos=None):
    return dates.num2date(x).strftime('%Y') #use FuncFormatter to format dates

# plot
fig = plt.figure()
plt.rcParams['grid.color'] = "black"
# plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.linestyle'] = '--'

ax = Axes3D(fig, azim = 35, elev = 25)
ax.set_facecolor("white")
ax.w_xaxis.set_pane_color((1,1,1,1))
ax.w_yaxis.set_pane_color((1,1,1,1))
ax.w_zaxis.set_pane_color((1,1,1,1))

# plot3D(df['maturity'], df.index, df['yield'],fig,ax,title=Yields in percentage')
mesh_plot2(df['maturity'], dates.date2num(df.index), df['yield'],fig,ax,title='Yields in percentage')
    

#%% extra

# # create dataframes Italy
# maturities_in_years = np.array([1,2,5,10,20,30])
# nom_yields_df = yields_df[['V13','V15','V17','V19','V21','V23']]
# nom_yields_df.columns = maturities_in_years
# real_yields_df = yields_df[['V14','V16','V18','V20','V22','V24']]
# real_yields_df.columns = maturities_in_years

# # Obtain summary statistics
# nom_stats = data_statistics(df=nom_yields_df, annualize=1)
# real_stats = data_statistics(df=real_yields_df, annualize=1)

# # # get prices
# # nom_prices_df = 1/np.power(1 + nom_yields_df/100, maturities_in_years)
# # real_prices_df = 1/np.power(1 + real_yields_df/100, maturities_in_years)

# # stats
# denom_high = nom_yields_df.mean() 
# dereal_high = real_yields_df.mean()
# frnom = nom_yields_df.mean()
# frreal = real_yields_df.mean()
# spnom = nom_yields_df.mean() 
# spreal = real_yields_df.mean()
# itnom_neg = nom_yields_df.mean() 
# itreal_neg = real_yields_df.mean()

