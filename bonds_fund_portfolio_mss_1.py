import os
path = os.path.dirname(os.path.realpath('__file__'))

import numpy as np
import pandas as pd
import seaborn as sns
sns.set(palette = 'bright',color_codes=True)
import pytz

from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
# from qstrader.alpha_model.nn_group_signals import NNGroupSignals

from qstrader.asset.equity import Equity
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession


if __name__ == "__main__":
    # forward test on simulated data buy and hold strategy
    # broker: substract liabilities each year
    
    start_dt = pd.Timestamp('2009-12-01 14:30:00', tz=pytz.UTC)
    end_dt = pd.Timestamp('2074-12-31 23:59:00', tz=pytz.UTC)
    
    # Construct the symbol and asset necessary for the backtest
    # strategy_symbols = ['SPY', 'TLT', 'IEI', 'SHY','TIP']
    strategy_symbols = ['INFMSS1DE1','INFMSS1DE2','INFMSS1DE5','INFMSS1DE10','INFMSS1DE20','INFMSS1DE30',
                        'NOMMSS1DE1','NOMMSS1DE2','NOMMSS1DE5','NOMMSS1DE10','NOMMSS1DE20','NOMMSS1DE30',
                        'INFMSS1IT1','INFMSS1IT2','INFMSS1IT5','INFMSS1IT10','INFMSS1IT20','INFMSS1IT30',
                        'NOMMSS1IT1','NOMMSS1IT2','NOMMSS1IT5','NOMMSS1IT10','NOMMSS1IT20','NOMMSS1IT30']
    
    strategy_assets = ['EQ:%s' % symbol for symbol in strategy_symbols]
    strategy_universe = StaticUniverse(strategy_assets)

    # To avoid loading all CSV files in the directory, set the
    # data source to load only those provided symbols
    DEFAULT_CONFIG_FILENAME = 'C:/Users/Eigenaar/OneDrive/QF/Seminar Financial Case Studies/advanced-algorithmic-trading-with-full-source-code/main dennis/nn group/yield data/data'
    csv_dir = os.environ.get('QSTRADER_CSV_DATA_DIR', DEFAULT_CONFIG_FILENAME)
    data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=strategy_symbols)
    data_handler = BacktestDataHandler(strategy_universe, data_sources=[data_source])
    
    # Read in NN Group's liabilities cashflows 
    cashflows_file = 'cashflows_low.csv' #!
    cashflows_df = pd.read_csv(os.path.join(path, cashflows_file), sep=',')
    dates = pd.date_range(start='2009-12-01', end='2074-12-31', freq='BY')
    cashflows_df.set_index(pd.to_datetime(dates,utc=True), inplace=True)
    cashflows_df.rename(lambda dt: dt.replace(hour=21, minute=0, second=0), inplace=True)

    # Construct an Alpha Model that simply provides a fixed
    # signal for the single GLD ETF at 100% allocation
    # with a backtest that does not rebalance
    strategy_alpha_model = FixedSignalsAlphaModel({'EQ:INFMSS1DE1': 0.35/4, 
                                                    'EQ:INFMSS1DE2': 0.4/4,
                                                    'EQ:INFMSS1DE5': 0.2/4,
                                                    'EQ:INFMSS1DE10': 0.02/4,
                                                    'EQ:INFMSS1DE20': 0.02/4,
                                                    'EQ:INFMSS1DE30': 0.01/4,
                                                    'EQ:NOMMSS1DE1': 0.35/4, 
                                                    'EQ:NOMMSS1DE2': 0.4/4,
                                                    'EQ:NOMMSS1DE5': 0.2/4,
                                                    'EQ:NOMMSS1DE10': 0.02/4,
                                                    'EQ:NOMMSS1DE20': 0.02/4,
                                                    'EQ:NOMMSS1DE30': 0.01/4,
                                                    'EQ:INFMSS1IT1': 0.35/4, 
                                                    'EQ:INFMSS1IT2': 0.4/4,
                                                    'EQ:INFMSS1IT5': 0.2/4,
                                                    'EQ:INFMSS1IT10': 0.02/4,
                                                    'EQ:INFMSS1IT20': 0.02/4,
                                                    'EQ:INFMSS1IT30': 0.01/4,
                                                    'EQ:NOMMSS1IT1': 0.35/4, 
                                                    'EQ:NOMMSS1IT2': 0.4/4,
                                                    'EQ:NOMMSS1IT5': 0.2/4,
                                                    'EQ:NOMMSS1IT10': 0.02/4,
                                                    'EQ:NOMMSS1IT20': 0.02/4,
                                                    'EQ:NOMMSS1IT30': 0.01/4})
    
    # strategy_alpha_model = NNGroupSignals({'EQ:TLT': 1.0})
    strategy_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        strategy_universe,
        strategy_alpha_model,
        initial_cash=6e9,
        rebalance='end_of_year',
        long_only=True,
        cash_buffer_percentage=0.1,
        data_handler=data_handler,
        cashflows_df=cashflows_df
    )
    strategy_backtest.run()
    
    # # Construct benchmark assets (buy & hold SPY)
    # benchmark_symbols = ['SPY']
    # benchmark_assets = ['EQ:SPY']
    # benchmark_universe = StaticUniverse(benchmark_assets)
    # benchmark_data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=benchmark_symbols)
    # benchmark_data_handler = BacktestDataHandler(benchmark_universe, data_sources=[benchmark_data_source])

    # # Construct a benchmark Alpha Model that provides
    # # 100% static allocation to the SPY ETF, with no rebalance
    # benchmark_alpha_model = FixedSignalsAlphaModel({'EQ:SPY': 1.0})
    # benchmark_backtest = BacktestTradingSession(
    #     start_dt,
    #     end_dt,
    #     benchmark_universe,
    #     benchmark_alpha_model,
    #     rebalance='buy_and_hold',
    #     long_only=True,
    #     cash_buffer_percentage=0.01,
    #     data_handler=benchmark_data_handler
    # )
    # benchmark_backtest.run()
    
    # Performance Output
    tearsheet = TearsheetStatistics(
        strategy_equity=strategy_backtest.get_equity_curve(),
        # benchmark_equity=benchmark_backtest.get_equity_curve(),
        # title='US TLT/TIP 60/40 Mix ETF Strategy',
        title='Strategic Weight Mixed German/Italian Bonds Strategy',
        rolling_sharpe=False
    )
    tearsheet.plot_results()

#%%
import qstrader.statistics.performance as perf
strategy_equity=strategy_backtest.get_equity_curve()
stats = tearsheet.get_results(strategy_equity)

returns = stats["returns"]
cum_returns = stats['cum_returns']

cagr = perf.create_cagr(cum_returns)
sharpe = perf.create_sharpe_ratio(returns)
sortino = perf.create_sortino_ratio(returns)
dd, dd_max, dd_dur = perf.create_drawdowns(cum_returns)


df = pd.DataFrame(['{:.2%}'.format(cagr), 
                   '{:.2f}'.format(sharpe),
                   '{:.2f}'.format(sortino),
                   '{:.2%}'.format(returns.std() * np.sqrt(252)),
                   '{:.2%}'.format(dd_max),
                   '{:.0f}'.format(dd_dur)], index= ['CAGR', 
                                                     'Sharpe Ratio', 
                                                     'Sortino Ratio', 
                                                     'Annual Volatility',
                                                     'Max Daily Drawdown',
                                                     'Max Drawdown Duration (Days)'])

#%%                                                    
# strategy_alpha_model = FixedSignalsAlphaModel({'EQ:INFMSS1DE1': 0.35, 
#                                                 'EQ:INFMSS1DE2': 0.4,
#                                                 'EQ:INFMSS1DE5': 0.2,
#                                                 'EQ:INFMSS1DE10': 0.02,
#                                                 'EQ:INFMSS1DE20': 0.02,
#                                                 'EQ:INFMSS1DE30': 0.01})
# strategy_alpha_model = FixedSignalsAlphaModel({'EQ:NOMMSS1DE1': 0.35, 
#                                                 'EQ:NOMMSS1DE2': 0.4,
#                                                 'EQ:NOMMSS1DE5': 0.2,
#                                                 'EQ:NOMMSS1DE10': 0.02,
#                                                 'EQ:NOMMSS1DE20': 0.02,
#                                                 'EQ:NOMMSS1DE30': 0.01})
# strategy_alpha_model = FixedSignalsAlphaModel({'EQ:INFMSS1IT1': 0.35, 
#                                                 'EQ:INFMSS1IT2': 0.4,
#                                                 'EQ:INFMSS1IT5': 0.2,
#                                                 'EQ:INFMSS1IT10': 0.02,
#                                                 'EQ:INFMSS1IT20': 0.02,
#                                                 'EQ:INFMSS1IT30': 0.01})
# strategy_alpha_model = FixedSignalsAlphaModel({'EQ:NOMMSS1IT1': 0.35, 
#                                                 'EQ:NOMMSS1IT2': 0.4,
#                                                 'EQ:NOMMSS1IT5': 0.2,
#                                                 'EQ:NOMMSS1IT10': 0.02,
#                                                 'EQ:NOMMSS1IT20': 0.02,
#                                                 'EQ:NOMMSS1IT30': 0.01})
                   
#%%
# # rebalance schedule
# rebalance_schedule = strategy_backtest.rebalance_schedule

# # alpha model
# weights = strategy_backtest.alpha_model.signal_weights
# weight = strategy_backtest.alpha_model.weight
# weight = pd.DataFrame(weight).T

# # portfolio construction
# adj_close_returns = strategy_backtest.qts.portfolio_construction_model.optimiser.adj_close_returns
# weight = strategy_backtest.qts.portfolio_construction_model.optimiser.weight
# weight = pd.DataFrame(weight).T
 
# # pcm methods
# dt = pd.Timestamp('2022-12-30 21:00:00', tz=pytz.UTC)
# optimised_weights = strategy_backtest.qts.portfolio_construction_model.optimiser(dt, initial_weights=weight)
# full_assets = strategy_backtest.qts.portfolio_construction_model._obtain_full_asset_list(dt)
# full_zero_weights = strategy_backtest.qts.portfolio_construction_model._create_zero_target_weight_vector(full_assets)
# full_weights = strategy_backtest.qts.portfolio_construction_model._create_full_asset_weight_vector(full_zero_weights, optimised_weights)

# target_portfolio =  strategy_backtest.qts.portfolio_construction_model._generate_target_portfolio(dt, full_weights)
# current_portfolio = strategy_backtest.qts.portfolio_construction_model._obtain_current_portfolio()
# rebalance_orders = strategy_backtest.qts.portfolio_construction_model._generate_rebalance_orders(dt, target_portfolio, current_portfolio)

#%%
# # order sizer
# order_sizer = strategy_backtest.qts.portfolio_construction_model.order_sizer

# # execution handler
# strategy_backtest.qts.execution_handler(dt, rebalance_orders)


#%%
# # broker init
# account_name = 'Backtest Simulated Broker Account'
# portfolio_id = '000001'
# portfolio_name = 'Backtest Simulated Broker Portfolio'
# account_id = strategy_backtest.broker.account_id
# base_currency = strategy_backtest.broker.base_currency
# initial_funds = strategy_backtest.broker.initial_funds
# cash_balances = strategy_backtest.broker.cash_balances
# portfolios = strategy_backtest.broker.portfolios

# # broker methods
# strategy_backtest.broker.subscribe_funds_to_account(7)
# strategy_backtest.broker.withdraw_funds_from_account(4)
# strategy_backtest.broker.get_account_cash_balance()

# strategy_backtest.broker.get_account_total_market_value()
# strategy_backtest.broker.get_account_total_equity() # cash and assets
# strategy_backtest.broker.list_all_portfolios()

# # broker methods on portfolio
# strategy_backtest.broker.get_portfolio_as_dict(portfolio_id)
# strategy_backtest.broker.get_portfolio_cash_balance(portfolio_id)
# strategy_backtest.broker.get_portfolio_total_equity(portfolio_id)

# # portfolio init
# strategy_backtest.broker.list_all_portfolios()[0].name
# strategy_backtest.broker.list_all_portfolios()[0].cash
# history = strategy_backtest.broker.list_all_portfolios()[0].history
# strategy_backtest.broker.portfolios[portfolio_id].portfolio_to_dict()

# # position handler init
# strategy_backtest.broker.portfolios[portfolio_id].pos_handler

# strategy_backtest.broker.portfolios[portfolio_id].pos_handler.positions    

#%%
# # NN Signals
# signal_weights = {}
# cash_outflow = 80e6
# port_dic = order_sizer.broker.list_all_portfolios()[0].portfolio_to_dict()
# total_equity = order_sizer._obtain_broker_portfolio_total_equity() # both cash and assets

# if len(port_dic.keys()) == 0:
#             signal_weights['EQ:TLT'] = 1.0
# else:
#     for key in port_dic.keys():
#         if key=='EQ:TLT':
#             mkt_val = port_dic[key]['market_value']
#             signal_weights[key] = np.round((mkt_val-cash_outflow)/total_equity, 2)
#         else:
#             mkt_val = port_dic[key]['market_value']
#             signal_weights[key] = np.round(mkt_val/total_equity, 2)


#%%
# # dollar_weighted init
# cash_buffer_percentage = 0.01

# # dollar_weighted call
# total_equity = strategy_backtest.broker.get_portfolio_total_equity(portfolio_id)

# cash_buffered_total_equity = total_equity * (
#     1.0 - cash_buffer_percentage
# )

# # Ensure weight vector sums to unity
# normalised_weights = order_sizer._normalise_weights(weights)

# target_portfolio = {}
# for asset, weight in sorted(normalised_weights.items()):
#     pre_cost_dollar_weight = cash_buffered_total_equity * weight

#     # Estimate broker fees for this asset
#     est_quantity = 0  # TODO: Needs to be added for IB
#     est_costs = order_sizer.broker.fee_model.calc_total_cost(
#         asset, est_quantity, pre_cost_dollar_weight, broker=order_sizer.broker
#     )

#     # Calculate integral target asset quantity assuming broker costs
#     after_cost_dollar_weight = pre_cost_dollar_weight - est_costs
#     asset_price = order_sizer.data_handler.get_asset_latest_ask_price(
#         dt, asset
#     )

#     if np.isnan(asset_price):
#         raise ValueError(
#             'Asset price for "%s" at timestamp "%s" is Not-a-Number (NaN). '
#             'This can occur if the chosen backtest start date is earlier '
#             'than the first available price for a particular asset. Try '
#             'modifying the backtest start date and re-running.' % (asset, dt)
#         )

#     # TODO: Long only for the time being.
#     asset_quantity = int(
#         np.floor(after_cost_dollar_weight / asset_price)
#     )

#     # Add to the target portfolio
#     target_portfolio[asset] = {"quantity": asset_quantity}

#%%
# # invicti signals
# events = {'Date': ['2019-11-13 21:00:00',
#                             '2019-11-25 21:00:00',
#                             '2019-11-27 21:00:00',
#                             '2019-12-11 21:00:00',
#                             '2019-12-16 21:00:00',
#                             '2019-12-18 21:00:00',
#                             '2020-02-10 21:00:00',
#                             '2020-03-13 21:00:00',
#                             '2020-03-16 21:00:00',
#                             '2020-03-26 21:00:00',
#                             '2020-03-26 21:00:00',
#                             '2020-03-27 21:00:00',
#                             '2020-03-27 21:00:00',
#                             '2020-03-31 21:00:00',
#                             '2020-03-27 21:00:00',
#                             '2020-06-19 21:00:00'
#                             ],
#                   'Security': ['BABA',
#                                 'SPLG',
#                                 'ISFA.AS',
#                                 'AMRN',
#                                 'DIS',
#                                 'ATVI',
#                                 'VAR1.DE',
#                                 'DIS',
#                                 'FLOW.AS',
#                                 'ISFA.AS',
#                                 'SPLG',
#                                 'CSCO',
#                                 'PYPL',
#                                 'AMRN',
#                                 'SPXU',
#                                 'SPXU'
#                                 ],
#                   'Buy/Sell': ['BUY',
#                                 'BUY',
#                                 'BUY',
#                                 'BUY',
#                                 'BUY',
#                                 'BUY',
#                                 'BUY',
#                                 'SELL',
#                                 'BUY',
#                                 'SELL',
#                                 'SELL',
#                                 'BUY',
#                                 'BUY',
#                                 'SELL',
#                                 'BUY',
#                                 'SELL'
#                                 ],
#                   'Amount': ['761',
#                               '713',
#                               '712',
#                               '536',
#                               '601',
#                               '478',
#                               '673',
#                               '354',
#                               '548',
#                               '506',
#                               '576',
#                               '448',
#                               '458',
#                               '94',
#                               '606',
#                               '296'
#                               ]}

# events = pd.DataFrame(events)
        
# dt_weights = {}
# port_dic = broker.list_all_portfolios()[0].portfolio_to_dict()
# equity = broker.list_all_portfolios()[0].total_equity
# sum_weights = 0

# for keys in port_dic.keys():
#     mkt_val = port_dic[keys]['market_value']
#     weights = np.round(mkt_val/equity, 2)
#     dt_weights[keys] = weights
#     if (keys != 'EQ:SHV'):
#         sum_weights += weights

# for row_index, row in events.iterrows():
#     # print("dataframes", pd.Timestamp(row['Date'], tz=pytz.UTC))
#     if (dt == pd.Timestamp(row['Date'], tz=pytz.UTC)):
#         equity = broker.list_all_portfolios()[0].total_equity
#         if (row['Buy/Sell'] == 'BUY'):
#             symbol = ('EQ:' + row['Security'])
#             weights = np.round(float(row['Amount'])/equity, 2)
#             dt_weights[symbol] = weights
#             sum_weights += weights
#             print("Success!\n", symbol, dt_weights[symbol])
#         if (row['Buy/Sell'] == 'SELL'):
#             symbol = ('EQ:' + row['Security'])
#             weights = 0
#             dt_weights[symbol] = weights
#             print("Success!\n", symbol, dt_weights[symbol])

# dt_weights['EQ:SHV'] = max(0, 1 - sum_weights)
# print("\n \n \n PAY ATTENTION\n", dt_weights)

