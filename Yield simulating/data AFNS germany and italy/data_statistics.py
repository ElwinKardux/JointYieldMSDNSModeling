# data statistics

import os as os
import sys as sys
path = os.path.dirname(os.path.realpath('__file__'))
from scipy import stats
from statsmodels.tsa import stattools
import numpy as np
import pandas as pd


def data_statistics(df, annualize):
    x = stats.describe(df)
    statistics = {'Mean': annualize*x[2], 'Variance': annualize*x[3], 
                  'Skewness': x[4], 'Kurtosis': x[5]}

    df_stats = pd.DataFrame(statistics, columns = ['Mean', 'Variance', 
                                                   'Skewness', 'Kurtosis', 
                                                   'jb_value', 'p', 'ar1', 
                                                   'ar6', 'ar12'], index = df.columns)
    ar0 = 0
    for i,industry in enumerate(df.columns):
        df_stats['jb_value'].iat[i], df_stats['p'].iat[i] = stats.jarque_bera(df[industry])
        acf = stattools.acf(df[industry], fft=True, nlags=12)
        df_stats['ar1'].iat[i], df_stats['ar6'].iat[i], df_stats['ar12'].iat[i] = acf[1], acf[6], acf[12]
        
    return df_stats

