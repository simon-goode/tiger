import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

def moments(df, *, kurtosis_fisher=True):
    """produces dataframe with four statistical moments calculated for each continuous variable specified

    - 'mean': first moment
    - 'var': second central moment
    - 'skew': third standard moment
    - 'kurtosis': fourth standard moment, defaults to Fisher, otherwise Pearson's kurtosis is used
    """
    dfn = df.select_dtypes(include=np.number)
    dfs = pd.DataFrame(columns=['mean', 'var', 'skew', 'kurtosis'], index=dfn.columns)
    for c in dfn.columns:
        dfs.loc[c, 'mean'] = np.mean(dfn[c].dropna())
        dfs.loc[c, 'var'] = np.var(dfn[c].dropna(), ddof=1)
        dfs.loc[c, 'skew'] = sp.stats.skew(dfn[c].dropna())
        dfs.loc[c, 'kurtosis'] = sp.stats.kurtosis(dfn[c].dropna(), fisher=kurtosis_fisher)
    return dfs


def corr_map(df, *, font_size=12, ax=None):
    """produces heatmap for lower triangle of correlation matrix"""
    if ax is None:
        plt.gca()

    c = df.corr()
    m = np.zeros_like(c, dtype=np.bool)
    m[np.triu_indices_from(m)] = True

    heatplot = sns.heatmap(c, mask=m,
                           cmap='copper', cbar_kws={"shrink": .6},
                           annot=True, annot_kws={"size": font_size},
                           vmax=1, vmin=-1, linewidths=.5,
                           square=True, ax=ax)
    return heatplot.figure