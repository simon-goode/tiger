import numpy as np
import sys
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import itertools
__all__ = ['BlankEncoder', 'InterEncoder', 'get_df_cols_diff']


class _SuppressPrints:
    """
    internally supresses print output from statsmodels
    
    thanks to [appelpy](https://github.com/mfarragher/appelpy/tree/master) for this snippet!
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def _df_input_conditions(X, y):
    if (y.isin([np.inf, -np.inf]).any() or
        len(X[X.isin([np.inf, -np.inf]).any(axis=1)]) > 0):
        raise ValueError(
            '''remove infinite values from dataset before modeling''')
    if (len(X.select_dtypes(['category']).columns.tolist()) > 0
        or pd.api.types.is_categorical_dtype(y.dtype)):
        raise TypeError(
            '''encode blanks from Pandas Category column(s) before modeling'''
        )
    if (len(X.select_dtypes(['O']).columns.tolist()) > 0
        or pd.api.types.is_categorical_dtype(y.dtype)):
        raise TypeError(
            '''remove columns with string data before modeling'''
        )
    if y.isnull().values.any() or X.isnull().values.any():
        raise ValueError(
            '''remove observations with null values before modeling'''
        )
    pass


class BlankEncoder:
    """
    blank variables for categorical columns
    """
    def __init__(self, df, cat_cols_base, *, mode='row_zero', separator='_'):
        if separator == '#':
            raise ValueError("""'#' is reserved for interaction terms""")
        if mode not in ['row_of_zero', 'blank_for_nan', 'row_of_nan']:
            raise ValueError("the argument for mode is not one of [row_of_zero, blank_for_nan, row_of_nan]")
        
        self._df = df
        self._cat_cols_base = cat_cols_base
        self._mode = mode
        self._separator = separator
    
    @property
    def df(self):
        "pd.DataFrame: dataframe to use for encoding"
        return self._df
    
    @property
    def cat_cols_base(self):
        "dict: categorical columns paired with base"
        return self._cat_cols_base
    
    @property
    def mode(self):
        "str: mode used for encoding"
        return self._mode
    
    @property
    def separator(self):
        "str: separator character"
        return self._separator

    def transform(self):
        """encode categories into blank columns for a new dataframe

        returns:
            pd.DataFrame: dataframe with categorical variables encoded into blank columns
        """
        p = self._df.copy()

        for col in self._cat_cols_base.keys():
            b = self._cat_cols_base[col]
            if self._cat_cols_base[col] == min:
                b = min(self._df[col].dropna().unique())
                self._cat_cols_base[col] = b
            if self._cat_cols_base[col] == max:
                b = max(self._df[col].dropna().unique())
                self._cat_cols_base[col] = b
            
            # generate blanks
            if self._mode == 'row_of_zero':
                c = pd.get_dummies(self._df[col], prefix=col, prefix_sep=self._separator)
            if self._mode == 'blank_for_nan':
                if np.count_nonzero((~pd.isna(self._df[col].to_numpy())) == len(self._df[col].to_numpy())):
                    c = pd.get_dummies(self.df[col], prefix=col, prefix_sep=self._separator)
                else:
                    c = pd.get_dummies(self._df[col], prefix=col, prefix_sep=self._separator)
                    nan_str = ''.join([col, self._separator, 'nan'])
                    c[nan_str] = np.where(c.sum(axis='columns') == 0, 1, 0)
            if self._mode == 'row_of_nan':
                c = pd.get_dummies(self._df[col], prefix=col, prefix_sep=self._separator)
                if self._df[col].isna().any():
                    nan_i = list(c[((c == 0).all(axis='columns'))].index)
                    c.loc[nan_i] = np.NaN
                    c = c.astype(pd.Int64Type())
            
            if b is not None:
                b_str = ''.join([col, self._separator, str(b)])
                del c[b_str]
            
            p = pd.concat([p, c], axis='columns')

            del p[col]
        return p


class InterEncoder:
    """encode interaction effects between variables"""
    def __init__(self, df, interactions, *, separator='_'):
        self._df = df
        self._interactions = interactions
        self._separator = separator
    
    @property
    def df(self):
        "pd.DataFrame: dataframe to use for encoding"
        return self._df
    
    @property
    def interactions(self):
        "dict: columns of interaction"
        return self._interactions
    
    @property
    def separator(self):
        "str: separator character"
        return self._separator
    
    def transform(self):
        """encode interactions between variables

        returns:
            pd.DataFrame: dataframe with interaction effects appended
        """
        p = self._df.copy()

        for c1, col_list in self._interactions.items():
            for c2 in col_list:
                c1_dtype = self._df[c1].dtype.name
                c2_dtype = self._df[c2].dtype.name

                c1_bool = self._df[c1].isin([0, 1, np.NaN]).all()
                c2_bool = self._df[c2].isin([0, 1, np.NaN]).all()

                if c1_bool:
                    b1 = pd.Series(self._df[c1], name=c1)
                if c1_dtype == 'category':
                    b1 = pd.get_dummies(self._df[c1], prefix=str(c1))
                if c2_bool:
                    b2 = pd.Series(self._df[c2], name=c2)
                if c2_dtype == 'category':
                    b2 = pd.get_dummies(self._df[c2], prefix=str(c2))

                # 1) both bool
                # 2) both continuous
                # 3) both categorical
                # 4-5) bool, categorical
                # 6-7) bool, continuous
                # 8-9) categorical, continuous

                if c1_bool and c2_bool:
                    i = '#'.join(str(c) for c in (c1, c2))
                    ib = pd.Series(b1 * b2, name=i, dtype=pd.Int64Type())
                    p = pd.concat([p, ib], axis='columns')
                    continue
                if ((c1_dtype != 'category' and not c1_bool) and (c2_dtype != 'category' and not c2_bool)):
                    i = '#'.join(str(c) for c in (c1, c2))
                    p[i] = (p[c1] * p[c2])
                    continue

                if ((c1_bool or c1_dtype == 'category') and (c2_bool or c2_dtype == 'category')):
                    aux = pd.concat([b1, b2], axis='columns')
                ib_df = pd.DataFrame(index=self._df.index)
                i_l = []

                if c1_dtype == 'category' and c2_dtype == 'category':
                    for co in itertools.product(b1, b2):
                        cv1, cv2 = co
                        i = '#'.join(str(c) for c in co)
                        i_l.append(i)
                        ib_df[i] = (aux[cv1] * aux[cv2])
                    b = BlankEncoder(p, {c1: None, c2: None}, separator=self._separator)
                    p = b.transform()
                    p = pd.concat([p, ib_df], axis='columns')
                    continue

                if c1_bool and c2_dtype == 'category':
                    for co in itertools.product([c1], b2.columns.values()):
                        cv1, cv2 = co
                        i = '#'.join(str(c) for c in co)
                        i_l.append(i)
                        ib_df[i] = (b1 * aux[cv2])
                    b = BlankEncoder(p, {c2: None}, separator=self._separator)
                    p = b.transform()
                    p = pd.concat([p, ib_df], axis='columns')
                    continue

                if c1_dtype == 'category' and c2_bool:
                    for co in itertools.product(b1.columns.values(), [c2]):
                        cv1, cv2 = co
                        i = '#'.join(str(c) for c in co)
                        i_l.append(i)
                        ib_df[i] = (aux[cv1] * b2)
                    b = BlankEncoder(p, {c1: None}, separator=self._separator)
                    p = b.transform()
                    p = pd.concat([p, ib_df], axis='columns')
                    continue

                if c1_bool and c2_dtype != 'category':
                    i = '#'.join(str(c) for c in (c1, c2))
                    i_b = pd.Series(b1 * self._df[c2], name=i)
                    p = pd.concat([p, i_b], axis='columns')
                    continue

                if c1_dtype != 'category' and c2_bool:
                    i = '#'.join(str(c) for c in (c1, c2))
                    i_b = pd.Series(self._df[c1] * b2, name=i)
                    p = pd.concat([p, i_b], axis='columns')
                    continue

                if c1_dtype == 'category' and (c2_dtype != 'category' and not c2_bool):
                    for co in itertools.product(b1.columns.values(), [c2]):
                        cv1, cv2 = co
                        i = '#'.join(str(c) for c in (cv1, cv2))
                        ib_df[i] = (b1[cv1] * self._df[c2])
                    b = BlankEncoder(p, {c1: None}, separator=self._separator)
                    p = b.transform()
                    p = pd.concat([p, ib_df], axis='columns')
                    continue

                if (c1_dtype != 'category' and not c1_bool) and c2_dtype == 'category':
                    for co in itertools.product([c1], b2.columns.values()):
                        cv1, cv2 = co
                        i = '#'.join(str(c) for c in (cv1, cv2))
                        ib_df[i] = (self._df[c1] * b2[cv2])
                    b = BlankEncoder(p, {c2: None}, separator=self._separator)
                    p = b.transform()
                    p = pd.concat([p, ib_df], axis='columns')
                    continue
        return p
    
    
def get_df_cols_diff(df_minuend, df_subtrahend):
    if (not isinstance(df_minuend, pd.DataFrame) or (not isinstance(df_subtrahend, pd.DataFrame))):
        raise TypeError("one or more arguments not of type pd.DataFrame")
    return list(set(df_minuend.columns) - set(df_subtrahend.columns))
