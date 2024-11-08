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


def summary_statistics(df):
    """returns summary statistics of dataframe quartiles"""
    summary_df = df.describe(percentiles=[0.25, 0.5, 0.75]).T
    summary_df['variance'] = df.var()
    summary_df['skewness'] = df.skew()
    summary_df['kurtosis'] = df.kurt()
    return summary_df


def plot_distributions(df):
    """plots histogram distribution of columns of dataframe"""
    n = df.select_dtypes(include='number').columns
    nc = 3
    nr = (len(n) + nc - 1) // nc

    fig, axes = plt.subplots(nr, nc, figsize=(15, nr*5))
    axes = axes.flatten()

    for i, col in enumerate(n):
        sns.histplot(df[col], kde=True, ax=axes[i], color='orange')
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def missing_values_summary(df, show=True):
    """plots missing value counts by column, with option to hide figure and just return missing values df"""
    missing_df = df.isnull().sum()
    missing_df = missing_df[missing_df > 0].sort_values(ascending=False)

    if show:
        plt.figure(figsize=(10,6))
        sns.barplot(x=missing_df.index, y=missing_df.values, color='salmon')
        plt.title("Missing Values Count by Column")
        plt.xlabel("Columns")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    return missing_df