Main folder: get data, simulated yields and backtesting

Sub folders:
1. data AFNS germany and italy
- get_real_yields_with_nan: To obtain the complete monthly yield series from January 2000 to December 2020(252 observations)
- data_statistics: To get the summary statistics
- get_simulated_yields: Simulated yields as shown in the paper section 6.2

2. yield data
- get_funds_data: prepare yield data to feed into QSTrader

3. qstrader
- bonds_fund_portfolio_mss_1: main to run qstrader with cashflows_low
- Can be imported through Numpy.
