<h1 align="center">Tiger</h1>
<p align="center">
  <img src="assets/tiger_nobg.png" />
</p>

Tiger is an Applied Econometrics library for Python. Your one-stop shop for econometric modeling, regression analysis, model performance testing, and much more! This package streamlines econometric analysis by harnessing the raw power of Numpy and Pandas — while maintaining the breathe-easy syntax of R and Stata.

### Example
```python
from tiger.linear import OLS
model = OLS(df, ['income'], ['age', 'gen', 'yoe']).fit()
model.summary
```
<details>
  <summary>Output:</summary>

  ```
                            OLS Regression Results
==============================================================================
Dep. Variable:                 income   R-squared:                       0.992
Model:                            OLS   Adj. R-squared:                  0.966
Method:                 Least Squares   F-statistic:                     39.13
Date:                Sun, 10 Nov 2024   Prob (F-statistic):              0.117
Time:                        01:27:02   Log-Likelihood:                -47.390
No. Observations:                   5   AIC:                             102.8
Df Residuals:                       1   BIC:                             101.2
Df Model:                           3
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const      -4.875e+04   2.55e+04     -1.914      0.306   -3.72e+05    2.75e+05
age         2916.6667    721.688      4.041      0.154   -6253.247    1.21e+04
gen        -1.667e+04   8660.254     -1.925      0.305   -1.27e+05    9.34e+04
yoe          833.3333   3145.764      0.265      0.835   -3.91e+04    4.08e+04
==============================================================================
Omnibus:                          nan   Durbin-Watson:                   2.531
Prob(Omnibus):                    nan   Jarque-Bera (JB):                0.665
Skew:                          -0.222   Prob(JB):                        0.717
Kurtosis:                       1.270   Cond. No.                         320.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
</details>
