# get yield data

import os as os
import sys as sys 
path = os.path.dirname(os.path.realpath('__file__'))
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

def get_funds_data(country, bond_type, coverage, number):
    # set seed and read in market calendar dates (implement in qstrader)
    ID = 447283
    np.random.seed(ID)
    nyse = mcal.get_calendar('NYSE')
    nyse.valid_days(start_date='2009-12-01', end_date='2075-01-01')
    schedule = nyse.schedule(start_date='2009-12-01', end_date='2075-01-01')
    
    # read in monthly yields
    yields_file = country + bond_type + '_' + coverage +  '_' + number + '.csv'
    yields_df = pd.read_csv(os.path.join(path, yields_file), sep=',')
    yields_df.set_index(pd.date_range(start='2009-12-01', end='2075-01-01',freq='BM'), inplace=True)
    
    # concat monthly and daily data
    yields_daily_df = pd.DataFrame(np.nan, index=schedule.index, columns = yields_df.columns)
    yields_daily_df = pd.concat([yields_daily_df, yields_df], axis=0)
    yields_daily_df['index'] = yields_daily_df.index
    yields_daily_df.drop_duplicates(subset=['index'],keep='last', inplace=True)
    yields_daily_df.drop(columns=['index'], inplace=True)
    
    # interpolate data
    yields_daily_df.sort_index(inplace=True)
    yields_daily_df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
    yields_daily_df.interpolate(method='linear', limit_direction='backward', axis=0, inplace=True)
    
    # save data
    for i in yields_daily_df.columns:
        df = pd.DataFrame(yields_daily_df[i], columns = ['Open', 'High', 'Low', 'Close', 'Volume', i])
        df.index.name = 'Date'
        df.rename(columns={i: 'Adj Close'}, inplace=True)
        df.to_csv(path  + '/data/' + (bond_type + coverage + number + country).upper() + i + '.csv')

if __name__ == "__main__":

   countries = ['de', 'it']
   bond_types = ['nom', 'inf']
   coverages = ['mss']
   numbers = ['1', '2']
   
   for i in countries:
       for j in bond_types:
           for k in coverages:
               for l in numbers:
                   get_funds_data(i, j, k, l)

