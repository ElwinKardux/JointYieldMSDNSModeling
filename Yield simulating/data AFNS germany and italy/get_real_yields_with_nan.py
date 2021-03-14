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

# read in monthly nominal yields
yields_file = 'data/italy_yields.csv'
yields_df = pd.read_csv(os.path.join(path, yields_file), sep=',')
yields_df.set_index(pd.date_range(start='2000-01-01', end='2020-12-31',freq='BM'), inplace=True)
yields_df = yields_df[['1','2','5','10','20','30']]
yields_df.columns = maturities_in_years

# read in monthly bei
bei_file = 'data/bei.csv'
bei_df = pd.read_csv(os.path.join(path, bei_file), sep=',')
bei_df.set_index(pd.date_range(start='2000-01-01', end='2020-12-31',freq='BM'), inplace=True)
bei_df = bei_df[['1','2','5','10','20','30']]
bei_df.columns = maturities_in_years

# interpolate data
yields_df.interpolate(method='linear', limit_direction='backward', axis=1, inplace=True)
bei_df.interpolate(method='linear', limit_direction='backward', axis=0, inplace=True)

# create monthly real yields
real_yields_df = yields_df - bei_df

# Obtain summary statistics
nom_stats = data_statistics(df=yields_df, annualize=1)
real_stats = data_statistics(df=real_yields_df, annualize=1)

# # stats of difference yields (interpolated)
# denom = yields_df.copy()
# dereal = real_yields_df.copy()
# itnom = yields_df.copy()
# itreal = real_yields_df.copy()

# nomstd = (itnom - denom).std().T
# infstd = (itreal - dereal).std().T

# # plot data
# yields_df.plot()
# plt.xlabel("Date")
# plt.ylabel("Yield in percentage")
# # plt.title("Germany nominal yields for different maturities")

# real_yields_df.plot()
# plt.xlabel("Date")
# plt.ylabel("Yield in percentage")
# # plt.title("Germany real yields for different maturities")

# bei_df.plot()
# plt.xlabel("Date")
# plt.ylabel("Yield in percentage")
# plt.title("Breakeven inflation for different maturities")

# # save data for AFNS
# yields_df.to_csv(path  + '/data AFNS germany/itnom interpolated' + '.csv')
# real_yields_df.to_csv(path  + '/data AFNS germany/itinf interpolated' + '.csv')

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
   ax.plot_surface(XX,YY,ZZ,cmap=cm.coolwarm)
   
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
    df1 = pd.DataFrame(real_yields_df[i])
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

ax = Axes3D(fig, azim = 45, elev = 10)
ax.set_facecolor("white")
ax.w_xaxis.set_pane_color((1,1,1,1))
ax.w_yaxis.set_pane_color((1,1,1,1))
ax.w_zaxis.set_pane_color((1,1,1,1))

# plot3D(df['maturity'], df.index, df['yield'],fig,ax,title=Yields in percentage')
mesh_plot2(df['maturity'], dates.date2num(df.index), df['yield'],fig,ax,title='Yields in percentage')
    
