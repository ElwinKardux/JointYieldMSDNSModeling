# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 13:29:38 2021

@author: elwin
"""

import numpy as np
import pandas as pd
import utils as utils
from scipy import stats
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

# =============================================================================
# 
# TODO: Make m_i slightly larger, could do through Correction term. 
# TODO: Check whether each of the values in the final dataframe 'df' are in the correct corresponding year, and not some lagged year. 
# 
# =============================================================================

data_ECB = pd.read_excel(r'ECB_data.xlsx', header=None)
ECB = data_ECB.iloc[:,1]/100 + 1
ECB2 = ECB.cumprod()

data_infexp2021 = pd.read_excel(r'CF pattern EUR 2021 inflation assignment.xlsx', 'Sheet2')
inf_r = data_infexp2021.iloc[0,:]
mat = np.array([1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,100])
cs = CubicSpline(mat, inf_r)
curve_points = np.linspace(min(mat),65, 65)
bei = cs(curve_points) + 1
bei_df = pd.DataFrame({'Best Estimate Inflation': bei})
bei_cp = bei_df.cumprod()

nominal_Inflation_Rate = 0.012 # <- Best Estimate Inflation
nominal_forward_rate = -0.006 # <- Short term forward Rate
real_Inflation_Rate = 0.012 # <- Same as BEI?
real_forward_rate = 0.01 - 0.012 # <- forward rate - inflation
utils.global_constants(a = nominal_Inflation_Rate, b = nominal_forward_rate, c= real_Inflation_Rate, d =real_forward_rate)

NominalMarketPrice_t = 1
RealMarketPrice_t = 1

arr = np.arange(65)
index = 1
# NominalMarketPrice_T = index * np.power(1+0.025,arr) # ECB RATE (yields)
# RealMarketPrice2_T = index * np.power(1+0.02,arr) # Discounted by BEI
NominalMarketPrice_T = (data_ECB.iloc[:,1]/100 + 1).cumprod()
RealMarketPrice_T = ((1 + data_ECB.iloc[:,1]/100) -  bei_df['Best Estimate Inflation'] + 1 ).cumprod()

''' values below are estimates '''
t = 0  # 2020
T = np.arange(0,66)  # maturities!  
T_i = T[1]
T_imin = T[0]

a_n = 0.0001              # <- result of calibration to ATM caps (Hull & White) Done!
a_r = 0.05086             # <- Result to calibration to inflation rates (JY Model)

sigma_n = 0.0286          # <- result of calibration to ATM caps (Hull & White) Done!
sigma_I= 0.00884          # <- Based on Historical data of HICPxT
sigma_r = 0.0184         # <- Historical data based on germany yield curve

rho_nr = 0.87            # <- Using historical data
rho_rI = -0.02           # <- Using historical data
rho_In = 0.48           # <- Using historical data



#Vi_squared = utils.comp_Vi_squared(t,T_imin,  T_i, a_n, a_r, sigma_n, sigma_I, sigma_r, rho_nr, rho_rI)

cf_0 = utils.get_Cashflows(aggregate=1)  # sum of future values
cf_1 = utils.get_Cashflows(aggregate=0).astype(float) # original cash flows


t = 0

P_n = np.zeros(65)
m_ = np.zeros(65)
Vi_squared_  = np.zeros(65)

prices_caplets_long0 = np.zeros(65)
prices_caplets_short3 = np.zeros(65)
prices_caplets_long3 = np.zeros(65)
prices_caplets_short6 = np.zeros(65)
prices_floorlets_long0 = np.zeros(65)
prices_floorlets_shortmin2 = np.zeros(65)

# Non-Generous policy
K0 = 1.00
K3 = 1.05
K6 = 1.06
Kmin2 = 0.98

cf_1 = cf_1[1:]
flag = 0
for cashflow in cf_1:
    if (flag==1):
        Notional_2 = Notional
    Notional = cashflow # previous notional SUM


    for i in range(1,len(T)-1):
        ''' Still need to connect the right timestamps, for now the first period has cash flow 0, and i starts at value 1 (so that T_imin exists = T[0] = 0)'''
        ''' Notional =1 gives quick comparison of the caplet prices '''
        
        T_i = T[i]
        T_imin = T[i-1]
        NominalMarketPrice_T_imin = NominalMarketPrice_T[i-1] # Bond price b4 (nom) <- We are not considering compounding effect
        NominalMarketPrice_T_i = NominalMarketPrice_T[i]      # Bond price now (nom)
        RealMarketPrice_T_imin = NominalMarketPrice_T[i-1]
        RealMarketPrice_T_i = NominalMarketPrice_T[i]
        phi_i = 1 # fixed-leg year fraction. One

        
        P_n[i] = utils.comp_Pn(t, T_i, NominalMarketPrice_T[i], NominalMarketPrice_t, sigma_n, a_n)
        
        m_[i] = utils.comp_m_i(t, 0, T_i, NominalMarketPrice_T_imin, NominalMarketPrice_T_i, NominalMarketPrice_t, RealMarketPrice_T_imin, RealMarketPrice_T_i, RealMarketPrice_t,  a_n, a_r, sigma_n, sigma_I, sigma_r, rho_nr, rho_rI)
        
        Vi_squared_[i] = utils.comp_Vi_squared(t,T_imin,  T_i, a_n, a_r, sigma_n, sigma_I, sigma_r, rho_nr, rho_rI)
        Vi_squared_[i] = Vi_squared_[i] * 0.001
        
        prices_caplets_long0[i] += utils.comp_IICplt(t, T_imin, T_i,P_n[i], m_[i], Vi_squared_[i], K0, phi_i, Notional)
        prices_caplets_short3[i] += utils.comp_IICplt(t, T_imin, T_i, P_n[i], m_[i], Vi_squared_[i], K3, phi_i, Notional)
        
        prices_floorlets_long0[i] += utils.comp_IICplt(t,T_imin, T_i, P_n[i], m_[i], Vi_squared_[i], K0, phi_i, Notional, Call = False)
        prices_floorlets_shortmin2[i] += utils.comp_IICplt(t,T_imin, T_i, P_n[i], m_[i], Vi_squared_[i], Kmin2, phi_i, Notional, Call = False)
        
        if (flag==1):    
            prices_caplets_long3[i] += utils.comp_IICplt(t, T_imin, T_i,P_n[i], m_[i], Vi_squared_[i], K3, phi_i, Notional_2)
            prices_caplets_short6[i] += utils.comp_IICplt(t, T_imin, T_i, P_n[i], m_[i], Vi_squared_[i], K6, phi_i, Notional_2)
    
    flag = 1


# =============================================================================
# # Cap + Floor
# =============================================================================
prices_caplets_long0_disc = utils.discount_values_simple(prices_caplets_long0, 0.0)
prices_caplets_short3_disc = utils.discount_values_simple(prices_caplets_short3, 0.0)
BS1_caplets = prices_caplets_long0_disc - prices_caplets_short3_disc  # 1 billion?

YoYCapPrize_long0 = np.sum(prices_caplets_long0_disc) #/1_000_000
YoYCapPrize_short3 = np.sum(prices_caplets_short3_disc)# /1_000_000
BullSpread1 = YoYCapPrize_long0 - YoYCapPrize_short3  # 1 billion?

# =============================================================================
# # Recovery Clause
# =============================================================================
prices_caplets_long3_disc = utils.discount_values_simple(prices_caplets_long3, 0.000)
prices_caplets_short6_disc = utils.discount_values_simple(prices_caplets_short6, 0.00)
BS2_caplets = prices_caplets_long3_disc - prices_caplets_short6_disc  # 1 billion?

YoYCapPrize_long3 = np.sum(prices_caplets_long3_disc)#/1_000_000
YoYCapPrize_short6 = np.sum(prices_caplets_short6_disc)#/1_000_000
BullSpread2 = YoYCapPrize_long3 - YoYCapPrize_short6  # 1 billion?

# =============================================================================
# # Catch up Clause
# =============================================================================
prices_floorlets_long0_disc = utils.discount_values_simple(prices_floorlets_long0, 0.02)
prices_floorlets_shortmin2_disc = utils.discount_values_simple(prices_floorlets_shortmin2, 0.02)
Floorlets = prices_floorlets_long0_disc - prices_floorlets_shortmin2_disc  # 1 billion?

YoYFloorPrize_long0 =  np.sum(prices_floorlets_long0_disc)#/1_000_000
YoYFloorPrize_shortmin2 =  np.sum(prices_floorlets_shortmin2_disc)#/1_000_000
BullSpread3 = YoYFloorPrize_long0 - YoYFloorPrize_shortmin2  # 1 billion?

Final_price =  BullSpread1 + BullSpread2 - BullSpread3
Final_pricelets= BS1_caplets + BS2_caplets - Floorlets

''' Create dictionary that summarizes results! '''
d = {'Year': np.arange(1,66), 'cashflows': cf_1, 'Infl_clause': Final_pricelets, 'Cap_Floor': BS1_caplets, 'Recovery_clause': BS2_caplets, 'Catch-up clause':Floorlets}
df = pd.DataFrame(data=d)
df2 = df.loc[:, df.columns != 'Year'].sum(numeric_only=True)




''' Liabilitity Excel sheet containing the two dataframes! check in files folder. '''
with pd.ExcelWriter('Liabilities_high.xlsx') as writer:  
    df.to_excel(writer, sheet_name = "Liability")
    df2.to_excel(writer,sheet_name='Liability',startrow=1 , startcol=10)   


    