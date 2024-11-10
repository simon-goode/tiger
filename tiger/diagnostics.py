import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
import seaborn as sns
import matplotlib.pyplot as plt
from .discrete import Logit

__all__ = ['plot_residuals_vs_predictors', 'plot_residuals_vs_fitted', 'pp_plot', 'qq_plot', 'heteroskedasticity_test', 'partial_regression_plot']

def plot_residuals_vs_fitted(residual_values, fitted_values, *, ax=None):
    """plot a model's residual values on the fitted values
    
    useful for determining if linearity holds for a model"""
    if ax is None:
        ax = plt.gca()
    cplot = sns.regplot(x=fitted_values, y=residual_values, ax=ax, fit_reg=False)
    fig = cplot.figure
    ax.grid(True, linewith=0.5)
    ax.set_title("Residuals vs. Fitted Values")
    ax.set_xlabel(r"Fitted Values ($\^{y}$)")
    ax.set_ylabel("Residuals")
    return fig


def plot_residuals_vs_predictors(model, *, predictor, ax=None):
    """plot a model's residual values on the regressor values"""
    if predictor not in model.X_list:
        raise ValueError("predictor not found in model")
    if ax is None:
        ax = plt.gca()
    cplot = sns.regplot(x=model.df.loc[:, predictor], y=model.results.resid, ax=ax, fit_reg=False)
    fig = cplot.figure
    ax.grid(True, linewidth=0.5)
    ax.set_title("Residuals vs Regressor Values\n".format(predictor))
    ax.set_xlabel("{}".format(predictor))
    ax.set_ylabel("Residuals")
    return fig


def pp_plot(residual_values, *, ax=None):
    """compares empirical cumulative distribution against theoretical cumulative distribution"""
    if ax is None:
        ax = plt.gca()
    pplot = sm.ProbPlot(residual_values, fit=True)
    fig = pplot.ppplot(ax=ax, color='tab:orange', markersize=4, line='45')
    ax.grid(True, linewidth=0.5)
    ax.set_title("P-P Plot of Residuals")
    return fig


def qq_plot(residual_values, *, ax=None):
    """compares quantiles of empirical distribution against quantiles of theoretical distribution"""
    if ax is None:
        ax = plt.gca()
    pplot = sm.ProbPlot(residual_values, fit=True)
    fig = pplot.qqplot(ax=ax, line='s', color='tab:orange', markersize=4)
    ax.grid(True, linewidth=0.5)
    ax.set_title("Q-Q Plot of Residuals")
    return fig


def partial_regression_plot(model, df, regressor, *, annotate_results=False, ax=None):
    """shows the effect of adding another regressor to a regression model"""
    X_list = model.X.columns.tolist()
    y_list = [model.y.name]

    if ax is None:
        ax = plt.gca()
    
    if (regressor not in X_list and regressor in df.columns and not df[regressor].isnull().any()):
        model_ylist = (sm.OLS(df[y_list], sm.add_constant(df[X_list])).fit(disp=0))
        model_var = (sm.OLS(df[regressor], sm.add_constant(df[X_list])).fit(disp=0))
        cplot = sns.regplot(x=model_var.resid, y=model_ylist.resid, ci=None, truncate=True, line_kws={'color': 'red'})
        
        if annotate_results:
            model_full = (sm.OLS(model_ylist.resid, sm.add_constant(df[X_list + [regressor]])).fit(disp=0))
            ax.set_title(('Partial regression plot: {}\n(b={:.4f}, t={:.3f})'.format(regressor, model_full.params.loc[regressor], model_full.tvalues.loc[regressor])))
        else:
            ax.set_title('Partial regression plot: {}'.format(regressor))
        fig = cplot.figure
        ax.grid()
        ax.set_ylabel('e({} | X)'.format(y_list[0]))
        ax.set_xlabel('e({} | X)'.format(regressor))
        return fig
    elif (regressor not in X_list and regressor in df.columns and df[regressor].isnull().any()):
        raise ValueError("null values found in the regressor column")
    elif regressor in X_list:
        cols = list(set(X_list)) - set([regressor])
        model_ylist = (sm.OLS(df[y_list], sm.add_constant(df[cols])).fit(disp=0))
        model_var = (sm.OLS(df[regressor], sm.add_constant(df[cols])).fit(disp=0))
        cplot = sns.regplot(x=model_var.resid, y=model_ylist.resid, ci=None, trucate=True, line_kws={'color': 'red'})
        if annotate_results:
            ax.set_title(('Partial regression plot: {}\n(b={:.4f}, t={:.3f})'
                          .format(regressor,
                                  model.params.loc[regressor],
                                  model.tvalues.loc[regressor])))
        else:
            ax.set_title('Partial regression plot: {}'.format(regressor))
        fig = cplot.figure
        ax.grid()
        ax.set_ylabel('e({} | X)'.format(y_list[0]))
        ax.set_xlabel('e({} | X)'.format(regressor))
        return fig
    else:
        raise ValueError("regressor not found in dataset")
    

def heteroskedasticity_test(test_name, model, *, regressors_subset=None):
    """returns results of a heteroskedasticity test
    
    supported tests:
        - 'breusch_pagan': Stata's `hettest` command
        - 'breusch_pagan_studentized': R's `bptest` command
        - 'white': Stata's `imtest, white` command
    """
    test_summary = {'distribution': 'chi2'}

    if test_name == 'breusch_pagan':
        if regressors_subset:
            if not set(regressors_subset).issubset(set(model.X_list)):
                raise ValueError("regressor(s) not in dataset")
            reduced = sm.OLS(model.y, sm.add_constant(model.X[regressors_subset]))
            reduced_results = reduced.fit()
            sq_resid = (reduced_results.resid ** 2).to_numpy()
        else:
            sq_resid = (model.resid ** 2).to_numpy()

        scaled_sq_resid = sq_resid / sq_resid.mean()
        y_hat = model.results.fittedvalues.to_numpy()

        aux = sm.OLS(scaled_sq_resid, sm.add_constant(y_hat)).fit()

        test_summary['nu'] = 1
        test_summary['test_stat'] = aux.model.ess / 2
        test_summary['p_value'] = sp.stats.chi2.sf(test_summary['test_stat'], test_summary['nu'])
    elif test_name == 'breusch_pagan_studentized':
        if regressors_subset:
            if not set(regressors_subset).issubset(set(model.X_list)):
                raise ValueError(
                    'Regressor(s) not recognised in dataset.  Check the list given to the function.')
            reduced = sm.OLS(model.y,
                                   sm.add_constant(model.X[regressors_subset]))
            reduced_results = reduced.fit()
            test_summary['nu'] = 1
            stat, pval, _, _ = sms.het_breuschpagan(reduced_results.resid, reduced_results.model.exog)
            test_summary['test_stat'], test_summary['p_value'] = stat, pval
        else:
            test_summary['nu'] = 1
            stat, pval, _, _ = sms.het_breuschpagan(model.results.resid, model.results.model.exog)
            test_summary['test_stat'], test_summary['p_value'] = stat, pval
    elif test_name == 'white':
        test_summary['nu'] = (
            int((len(model.X_list) ** 2
                + 3 * len(model.X_list))
                / 2))
        white = sms.het_white(model.resid,
                                   sm.add_constant(model.X))
        test_summary['test_stat'] = white[0]
        test_summary['p_value'] = white[1]
    else:
        raise ValueError(
            """test_name not one of 'breusch_pagan', 'breusch_pagan_studentized' or
            'white'""")

    return test_summary


def variance_inflation_factors(X, *, vif_threshold=10):
    """Returns a DataFrame with variance inflation factor (VIF) values
    calculated given a matrix of regressor values (X).

    VIF values are used as indicators for multicollinearity.
    Econometrics literature typically uses a threshold of 10 to indicate
    problems with multicollinearity in a model.

    Args:
        X (pd.DataFrame): matrix of values for the regressors (one
            row per regressor).
        vif_threshold (int, optional): Defaults to 10.  The threshold
            set for assessment of multicollinearity.

    Returns:
        pd.DataFrame: columns for VIF values, their tolerance and the
            result of the heuristic.
    """
    X = sm.add_constant(X)
    # Set up variance inflation factor values:
    vif = pd.Series([1 / (1.0 - (sm.OLS(X[col].values,
                                        X.drop(columns=[col]).values))
                          .fit().rsquared)
                     for col in X],
                    index=X.columns,
                    name='VIF')
    vif = vif.drop('const')  # constant not needed for output
    # Calculate tolerance:
    tol = pd.Series(1 / vif, index=vif.index, name='1/VIF')
    vif_thres = pd.Series(vif > vif_threshold,
                          name="VIF>" + str(vif_threshold))
    # Final dataframe:
    return pd.concat([vif, tol, vif_thres], axis='columns')


def akaike_ic(model):
    """returns the Akaike Information Criterion (AIC) of the model"""
    n = len(model.resid)
    k = len(model.X_list)+1

    rss = np.sum(model.resid ** 2)
    if isinstance(model, Logit):
        llh = model.log_likelihood
    else:
        llh = -n / 2 * np.log(rss / n)

    aic = 2 * k - 2 * llh
    return aic


def bayesian_ic(model):
    """returns the Bayesian Information Criterion (AIC) of a model"""
    n = len(model.resid)
    k = len(model.X_list)+1

    rss = np.sum(model.resid ** 2)
    if isinstance(model, Logit):
        llh = model.log_likelihood
    else:
        llh = -n / 2 * np.log(rss / n)

    bic = k * np.log(n) - 2 * llh
    return bic


def mean_errors(model):
    """returns:
        - Mean Squared Error (MSE)
        - Mean Absolute Error (MAE)
        - Root Mean Squared Error (RMSE)"""
    mse = np.mean(model.resid ** 2)
    mae = np.mean(np.abs(model.resid))
    rmse = np.sqrt(mse)

    return mse, mae, rmse