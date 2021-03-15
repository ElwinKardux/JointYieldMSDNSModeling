# JointYieldMSDNSModeling
Joint modeling of the nominal and real yield using Markov-Switching Dynamic Nelson-Siegel models

* **main.m:**
Main file which initializes all the parameters, optimizes the log-likelihoods of the JDNS, the MSL-JDNS and the MSS-JDNS. Also provides code for the smoothed factor plots.

* **NegLogLike.m:**
Function of the log-likelihood for the JDNS.

* **NegLogLikeLambda.m:**
Function of the log-likelihood for the MSL-JDNS.

* **NegLogLikeSigmas.m:**
Function of the log-likelihood for the MSS-JDNS.

* **DNS.m:**
Function that takes in the JDNS parameters ($f$, $\alpha$, $\lambda$, $p$, $q$, $\mu$) and returns the state-space formulation parameters ($H, F, R, Q, d_t$).

* **Loadings\_Joint.m:**
Function that inputs $\alpha$, $t$ and $\lambda$, and returns the Nelson-Siegel factors loadings ($H$).

* **V\_Prob.m:**
Multinomial Normal log-likelihood function, with inputs: forecast error and its variance.

* **Liabilities/utils.py:**
File containing helper functions that we use to price the liabilities via the JY model.

* **Liabilities/main\_derivative\_pricing.py:**
Running this file will provide the values of the liabilities for all years corresponding to the liabilities of NN Leven. The methodology behind the tuning of this model is given in \autoref{Appendix: Option pricing}.

* **Liabilities/main\_hull\_white\_model\_v2.py**
In this file, we run the Hull-White model which provides the $a_n$ term corresponding to the mean reversion of the inflation curve.

* **Yield\_Simulating/bonds\_fund\_portfolio\_mss\_1.py:**
Creates the investment strategy and obtains the results of Table 5.


If you want to run the **Yield\_Simulating/bonds\_fund\_portfolio\_mss\_1.py** file, you have to import the QSTrader package first. The download link and the details on this package can be found in https://github.com/mhallsmoore/qstrader.
