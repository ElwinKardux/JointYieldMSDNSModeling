#%%
import QuantLib as ql
from collections import namedtuple
import math

#%%
today = ql.Date(1, ql.February, 2021);
settlement= ql.Date(28, ql.February, 2051);
ql.Settings.instance().evaluationDate = today;
term_structure = ql.YieldTermStructureHandle(
    ql.FlatForward(settlement,0.02,ql.Actual365Fixed())
    )
index = ql.Euribor1Y(term_structure)

CalibrationData = namedtuple("CalibrationData", 
                             "start, length, volatility")
data = [CalibrationData(1, 1, 4.00),
        CalibrationData(1, 2, 3.50),
        CalibrationData(1, 3, 4.00),
        CalibrationData(1, 4, 3.2),
        CalibrationData(1, 5, 3.5),
        CalibrationData(2, 1, 2.4),
        CalibrationData(2, 2, 3.8),
        CalibrationData(2, 3, 2.25),
        CalibrationData(2, 4, 2.40),
        CalibrationData(2, 5, 2.50),
        CalibrationData(5, 1, 1.75),
        CalibrationData(5, 2, 2.75),
        CalibrationData(5, 3, 2.50),
        CalibrationData(5, 4, 1.60),
        CalibrationData(5, 5, 1.75)]

#%%
def create_swaption_helpers(data, index, term_structure, engine):
    swaptions = []
    fixed_leg_tenor = ql.Period(1, ql.Years)
    fixed_leg_daycounter = ql.Actual360()
    floating_leg_daycounter = ql.Actual360()
    for d in data:
        vol_handle = ql.QuoteHandle(ql.SimpleQuote(d.volatility))
        helper = ql.SwaptionHelper(ql.Period(d.start, ql.Years),
                                   ql.Period(d.length, ql.Years),
                                   vol_handle,
                                   index,
                                   fixed_leg_tenor,
                                   fixed_leg_daycounter,
                                   floating_leg_daycounter,
                                   term_structure
                                   )
        helper.setPricingEngine(engine)
        swaptions.append(helper)
    return swaptions    
#%%
model = ql.HullWhite(term_structure);
engine = ql.JamshidianSwaptionEngine(model)
swaptions = create_swaption_helpers(data, index, term_structure, engine)

optimization_method = ql.LevenbergMarquardt(1.0e-8,1.0e-8,1.0e-8)
end_criteria = ql.EndCriteria(10000, 100, 1e-6, 1e-8, 1e-8)
model.calibrate(swaptions, optimization_method, end_criteria)

a, sigma = model.params()
print("a = {0:10.4f}, sigma = {1:10.4f}".format(a, sigma))

# %%
