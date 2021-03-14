# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:30:34 2021
@author: elwin
"""
import numpy as np
import pandas as pd
import xlrd
from scipy.stats import norm
from scipy.interpolate import interp1d
from openpyxl import load_workbook
import matplotlib.pyplot as plt
#import main_derivative_pricing as main
# from main_derivative_pricing import abc

def global_constants(a = 0.02, b = 0.03, c = 0.02, d= 0.02):
    """
    Function for globals of the 4 variables. This function is unneccesary but else most of the functions below would require 4 more input arguments,
    which can get quite ugly. Could also add global variables of a_n, b_n, sigma_n, etc. which would make the functions smaller as well, but I didn't bother.

    Parameters
    ----------
    a : nominal_Inflation_Rate
    b : nominal_forward_rate
    c : real_Inflation_Rate
    d : real_forward_rate

    Returns
    -------
    None.

    """
    global nominal_Inflation_Rate 
    nominal_Inflation_Rate = a 
    global nominal_forward_rate
    nominal_forward_rate = b
    global real_Inflation_Rate
    real_Inflation_Rate = c
    global real_forward_rate  # == f_r^M (0,t) 
    real_forward_rate = d
    
    
#%%    
"""
Pricing formulas for Hull and W hite model, Page 652 in Mercurio book. 
Could combine the functions of comp_An and comp_Ar, comp_Bn and comp_Br, and comp_Pn and comp_Pr.

""" 
def comp_An(t, maturity, NominalMarketPrice_T, NominalMarketPrice_t, sigma_n, a_n):#, nominal_forward_rate):
   
    Bn = comp_Bn(a_n, t, maturity)
    An = (NominalMarketPrice_T/NominalMarketPrice_t) * np.exp(Bn * nominal_forward_rate - (sigma_n * sigma_n/(4 * a_n)) * (1 -np. exp(-2 * a_n * t)) * Bn**2)
    return An

def comp_Bn(t, maturity, a_n):
    Bn = (1/a_n) * (1 -np.exp((-a_n) * (maturity - t)))
    return Bn

def comp_Ar(t, maturity,RealMarketPrice_T, RealMarketPrice_t, sigma_r, a_r): #,  real_forward_rate):
    Br = comp_Br(a_r, t, maturity)
    Ar = (RealMarketPrice_T/RealMarketPrice_t) * np.exp(Br * real_forward_rate - (sigma_r * sigma_r/(4 * a_r)) * (1- np.exp(-2 * a_r * t)) * Br**2)
    return Ar

def comp_Br(t, maturity, a_r):
    Br = (1/a_r) * (1 - np.exp((-a_r) * (maturity - t)))
    return Br

def comp_Pn(t, maturity, NominalMarketPrice_T, NominalMarketPrice_t, sigma_n, a_n):
    An = comp_An(t, maturity, NominalMarketPrice_T, NominalMarketPrice_t, sigma_n, a_n) #, nominal_forward_rate)
    Bn = comp_Bn(t, maturity, a_n)
    Pn = An * np.exp((-Bn) * nominal_Inflation_Rate)
    return Pn #, An, Bn

def comp_Pr(t, maturity, RealMarketPrice_T, RealMarketPrice_t, sigma_r, ar): #  real_forward_rate, real_Inflation_Rate):
    Ar = comp_Ar(t, maturity, RealMarketPrice_T, RealMarketPrice_t, sigma_r, ar)#,  real_forward_rate)
    Br = comp_Br(t, maturity, ar)
    ZC_RealBondPrice = Ar * np.exp((-Br) * real_Inflation_Rate)
    return ZC_RealBondPrice # Ar, Br



#%%
def comp_m_i(t,T_imin, T_i, NominalMarketPrice_T_imin, NominalMarketPrice_T_i, NominalMarketPrice_t, RealMarketPrice_T_imin, RealMarketPrice_T_i, RealMarketPrice_t,  a_n, a_r, sigma_n, sigma_I, sigma_r, rho_nr, rho_rI):
    """
    mean function, page 662. Important add on:  The book does not say anything, but the formula only works for t < T_{i-1}.  When t = T_{i-1}, you will get 
        values of infinity, hence I looked up more on it, and the book did say something about in the Swap pricing section, but not in the option pricing section.
        It came down to the adjustment in the if statement below, or page 653 Mercurio. 

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    T_imin : TYPE
        DESCRIPTION.
    T_i : TYPE
        DESCRIPTION.
    NominalMarketPrice_T_imin : TYPE
        DESCRIPTION.
    NominalMarketPrice_T_i : TYPE
        DESCRIPTION.
    NominalMarketPrice_t : TYPE
        DESCRIPTION.
    RealMarketPrice_T_imin : TYPE
        DESCRIPTION.
    RealMarketPrice_T_i : TYPE
        DESCRIPTION.
    RealMarketPrice_t : TYPE
        DESCRIPTION.
    a_n : TYPE
        DESCRIPTION.
    a_r : TYPE
        DESCRIPTION.
    sigma_n : TYPE
        DESCRIPTION.
    sigma_I : TYPE
        DESCRIPTION.
    sigma_r : TYPE
        DESCRIPTION.
    rho_nr : TYPE
        DESCRIPTION.
    rho_rI : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if T_imin == t:
        P_n_2 = comp_Pn(t,T_i, NominalMarketPrice_T_i, NominalMarketPrice_t, sigma_n, a_n)
        P_r_1 = comp_Pr(t,T_i, RealMarketPrice_T_i, RealMarketPrice_t, sigma_r, a_r)
        return P_r_1/P_n_2
    P_n_1 = comp_Pn(t,T_imin, NominalMarketPrice_T_imin, NominalMarketPrice_t, sigma_n, a_n)
    P_n_2 = comp_Pn(t,T_i, NominalMarketPrice_T_i, NominalMarketPrice_t, sigma_n, a_n)
    P_r_1 = comp_Pr(t,T_i, RealMarketPrice_T_i, RealMarketPrice_t, sigma_r, a_r)
    P_r_2 = comp_Pr(t,T_imin, RealMarketPrice_T_imin, RealMarketPrice_t, sigma_r, a_r)
    C = correction_term(t,T_imin, T_i, a_n, a_r, sigma_n, sigma_I, sigma_r, rho_nr, rho_rI)
    m_i = (P_n_1/ P_n_2)  *  (P_r_1/ P_r_2) * np.exp(C) + 0.01
    # print('P_n_1',P_n_1)
    # print('P_n_2',P_n_2)
    # print('P_r_1',P_r_1)
    # print('P_r_2',P_r_2)
    # print('C :' ,C)
    # print('m_i', m_i)
    
    return m_i

def correction_term(t,T_imin, T_i,  a_n, a_r, sigma_n, sigma_I, sigma_r, rho_nr, rho_rI):
    """
    Book page 653, correction term used for the Mean value and hence also the price of caplet.

    Parameters
    ----------
    t : TYPE
    T_imin : TYPE
    T_i : TYPE
    a_n : TYPE
    a_r : TYPE
    sigma_n : TYPE
    sigma_I : TYPE
    sigma_r : TYPE
    rho_nr : TYPE
    rho_rI : TYPE
    Returns
    -------
    C : TYPE
        Correction term C.
    """

    B_r_1 = comp_Br(T_imin,T_i, a_r)
    B_r_2 = comp_Br(t,T_imin, a_r)
    B_n_1 = comp_Bn(t,T_imin, a_n)
    part1 = sigma_r * B_r_1
    part2 = B_r_2* ( rho_rI  * sigma_I    - 0.5 * sigma_r * B_r_2 + (rho_nr*sigma_n/(a_n+a_r)) * (1+a_r * B_n_1)) -   rho_nr*sigma_n/(a_n+a_r) * B_n_1
    C = part1 * part2;
    return C


def comp_Vi_squared(t,T_imin,  T_i, a_n, a_r, sigma_n, sigma_I, sigma_r, rho_nr, rho_rI):
    """
    The variance value is very dependend on paramters. A a_n or a_r value larger than 0.5 usually results in variance of greater than 1 million.
    This value is very important for the caplet prices.

    Parameters
    ----------
    t : starting year
    T_imin : Previous payment date, in our case the year before maturity T
    T_i : Year of maturity
    a_n : mean reversion speed of nominal rate
    a_r : mean reversion speed of real rate
    sigma_n : volatility of nominal
    sigma_I : volatility of Inflation Index
    sigma_r : volatility of real rate
    rho_nr : correlation nominal, real rates
    rho_rI : correlation real, Inflation rates.

    Returns
    -------
    Vi_squared : The variance of the logarithm of the ratio can be equivalently calculated under
            the (nominal) risk-neutral measure. 

    """

    T_delta = T_i - T_imin;
    part1 = (sigma_n**2/(2*(a_n)**3))  * (1 - np.exp( -a_n *(T_delta)  ))**2   *( 1-np.exp(-2*a_n*(T_imin -t))) + sigma_I**2 * T_delta
    part2 =  (sigma_n**2/(2*(a_r)**3)) * (1 - np.exp( -a_r *(T_delta)  ))**2   *( 1-np.exp(-2*a_r*(T_imin -t)))
    part3 = - 2*rho_nr * sigma_n * sigma_r /(a_n*a_r*(a_n+a_r)) * (1- np.exp( -a_n *(T_delta))) * (1- np.exp( -a_r * (T_delta)))  * (1- np.exp( - (a_n+a_r )*(T_imin - t)))
    part4 = (sigma_n**2/(a_n**2))* (T_delta + (2/a_n)*np.exp(-a_n*T_delta) - (1/(2*a_n)) * np.exp(-2*a_n * T_delta) - (3/(2*a_n)) ) 
    # (sigma_n**2/(a_n**2))  = 484?? 
    part5 = (sigma_r**2/(a_r**2))* (T_delta + (2/a_r)*np.exp(-a_r*T_delta) - (1/(2*a_r)) * np.exp(-2*a_r * T_delta) - (3/(2*a_r)) )
    #part4 = part5 ???   of course as a_n = a_r for now.
    part6 = - 2* rho_nr  * (sigma_n * sigma_r / (a_n * a_r)) *(T_delta -  ((1-np.exp(-a_n * T_delta))/a_n)  - ((1-np.exp(-a_r * T_delta))/a_r))
    #part6  =748, huge!?
    part7 = - 2* rho_rI  * (sigma_r * sigma_I / (a_r)) *(T_delta - ((1-np.exp(-a_r * T_delta))/a_r));
    Vi_squared = part1 + part2 + part3 + part4 + part5 + part6 + part7
    return Vi_squared


def comp_IICplt(t, T_imin, T_i, P_n_i, m_i, Vi_squared_i, K, phi_i, Notional_i, Call = True):#, a_n, a_r, sigma_n, sigma_I, sigma_r, rho_nr, rho_rI):  
    """
    Page 663, equation 17.4 

    Parameters
    ----------
    t : starting year
    T_imin : Previous payment date, in our case the year before maturity T
    T_i : Year of maturity
    P_n_i: Prize of nominal bond prize:  P_n (t,T_i), as in the master paper
    Vi_squared_i :
    K : Strike price, I think it should be K =1.03 for 3% cap, so not like K=3. 
    phi_i: the contract year fraction for the interval [Ti−1, Ti], our case I guess its 1 year so =1
    Call = True:  If false, set w=-1 and get formula for IIfloorlet. 
    
    Returns
    -------
    Price of caplet, or floorlet when Call = False.  
    """
    
    #Pn = comp_Pn(NominalMarketPrice_T, NominalMarketPrice_t, NominalVolatility, a_n, t, maturity, nominal_forward_rate, nominal_Inflation_Rate);
    #m_i = comp_m_i(t,T_imin, T_i, a_n, a_r, sigma_n, sigma_I, sigma_r, rho_nr, rho_rI)
    w = 1
    if not Call:
        w = -1
    #print('pricing part 1:  ', norm.cdf(w*  (np.log(m_i/K)+ 0.5 * Vi_squared_i)/np.sqrt(Vi_squared_i) ))
    #print('pricing part 2:  ', -K * norm.cdf(w*  (np.log(m_i/K) - 0.5 * Vi_squared_i)/np.sqrt(Vi_squared_i)  ))
    price_IICplt = w * Notional_i * phi_i * P_n_i * ( m_i *  norm.cdf(w*  (np.log(m_i/K)+ 0.5 * Vi_squared_i)/np.sqrt(Vi_squared_i) )-K * norm.cdf(w*  (np.log(m_i/K) - 0.5 * Vi_squared_i)/np.sqrt(Vi_squared_i)  ))
    return price_IICplt 



#%%
def get_Cashflows(aggregate=0):
    """
    Importants excel file and gets cashflows.
    
    Returns
    -------
    cf : TYPE
    inf_r : TYPE
    mat : TYPE
    """
    
    data_cf = pd.read_excel(r'CF pattern EUR 2021 inflation assignment.xlsx')
    #data_infexp2021 = pd.read_excel(r'CF pattern EUR 2021 inflation assignment.xlsx', 'Sheet2')
    
    cf = data_cf.iloc[1:,1]
    cf = np.array(cf)
    # cf = np.pad(cf, (0,1))
    #inf_r = data_infexp2021.iloc[0,:]

    size = np.size(cf)
    agg_cf = np.zeros((size))
    for i in range(size):
        agg_cf[i] = np.sum(cf[i:])
    
    if (aggregate):
        cf_ret = agg_cf
    else:
        cf_ret = cf

    #mat = np.array([1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,100])
    return cf_ret #, inf_r, mat



def discount_values_simple(cf, disc_rate=0.02):
    """
    Discounts values

    Parameters
    ----------
    disc_rate : TYPE, optional
        DESCRIPTION. The default is 0.02.

    Returns
    -------
    disc_cf : discounted cashflow array, length: T - t, T = 66
    """
    size = np.size(cf)
    arr = np.arange(0,size)
    index = np.ones(size)
    indexation = index * np.power(1+disc_rate,arr)
    discounted = cf/indexation
    return discounted


def discount_values_curve(cf, disc_rate=0.02):
    """
    Discounts values

    Parameters
    ----------
    disc_rate : TYPE, optional
        DESCRIPTION. The default is 0.02.

    Returns
    -------
    disc_cf : discounted cashflow array, length: T - t, T = 66
    """

    arr = np.arange(0,66)
    index = np.ones(66)
    indexation = index * np.power(1+disc_rate,arr)
    discounted = cf/indexation
    return discounted

def get_discounted_cashflows(t = 0,disc_rate=0.02):
    """
    Obtains cashflows from year t, and discounts them to year t.

    Parameters
    ----------
    t : year, set equal to 0
    disc_rate : TYPE, optional
        DESCRIPTION. The default is 0.02.

    Returns
    -------
    disc_cf : discounted cashflow array, length: T - t, T = 66
    """
    disc_arr = np.zeros((65))
    for i in range(t, 65):
        cf, inf_r,mat = get_Cashflows()
        arr = np.arange(len(cf))    
        disc_cf = cf[i:] * np.power(1+disc_rate,-arr[i:])
        disc_arr[i] = np.sum(disc_cf)
    return disc_arr