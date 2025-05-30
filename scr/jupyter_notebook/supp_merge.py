import sys
import os
import set_path
import supp.support_load as sl

import numpy as np
import pandas as pd
import re


# This files defined 'merge' function for merging Maatbase dataframes
# To directly analyze colum-source dataframe the function
#    adds prefix to column names (except foreing/primary keys)
# The prefix is based on location od df in dfs list
#

# load data
dfs, dfs_name, dfs_export_date = sl.load_pickle()
iton, ntoi = sl.get_name_dicts(dfs_name)

# get prefix for columns
def merge_get_prefix(df):
    # Find the index of the matching DataFrame
    position = next((i for i, d in enumerate(dfs) if df.equals(d)), None)
    if position is None:
        return None
    return str(position).zfill(2)

# check whether given column is foreing/primary key
def contains_id(text):
    return bool(re.search(r'(^|_)ID(_|$)', text, re.IGNORECASE))

# add prefix to column name
def merge_add_prefix(df, prefix):
    df_new = df.add_prefix(f'{prefix}_', axis=1)
    df_new = df_new.rename(columns={f'{prefix}_{col}': col for col in df.columns if contains_id(col)})
    return df_new

# merge df1 and df2 by calling pandas.merge
def merge(df1, df2, on=None, left_on=None, right_on=None, how='left'):
    prefix1 = merge_get_prefix(df1)
    prefix2 = merge_get_prefix(df2)
    if prefix1 is not None:
        df1 = merge_add_prefix(df1, prefix1)
    if prefix2 is not None:
        df2 = merge_add_prefix(df2, prefix2)
    
    df = pd.merge(df1, df2, how=how,
                  on=on, left_on=left_on, right_on=right_on,
                  suffixes=(None, f'_{prefix2}')
                 )
    return df