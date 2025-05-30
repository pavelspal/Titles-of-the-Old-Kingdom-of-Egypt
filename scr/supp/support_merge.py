import sys
import os
import io
try:
    import supp.support_load as sl
except:
    import support_load as sl

import numpy as np
import pandas as pd
import re


# This files defined 'merge' function for merging Maatbase dataframes
# To directly analyze colum-source dataframe the function
#    adds prefix to column names (except foreing/primary keys)
# The prefix is based on location od df in dfs list
#

# ignore any print
original_stdout = sys.stdout  # Save original stdout
sys.stdout = io.StringIO()    # Redirect stdout to a dummy output
# load data
dfs, dfs_name, dfs_export_date = sl.load_pickle()
iton, ntoi = sl.get_name_dicts(dfs_name)
# Restore original stdout
sys.stdout = original_stdout 

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

# group columns into list by given column_id
def group_to_list(df, col_to_stay, col_to_list=None):
    if isinstance(col_to_stay, str):
        col_to_stay = [col_to_stay]
    if isinstance(col_to_list, str):
        col_to_list = [col_to_list]
    if col_to_list is None:
        col_to_list = [col for col in df.columns if col not in col_to_stay]

    df_result = df[col_to_stay].drop_duplicates()
    for col in col_to_list:
        df_new = df.groupby(col_to_stay)[col].apply(lambda x: list(set(x.dropna()))).reset_index()
        df_result = pd.merge(df_result, df_new, on=col_to_stay, how='left')

    # add prefix if necessary
    prefix = merge_get_prefix(df)
    if prefix is not None:
        df_result = merge_add_prefix(df_result, prefix)
        # rename columns
        df_result.rename(columns={f'{prefix}_{col}': f'{prefix}_{col}_list' for col in col_to_list}, inplace=True)
    
    # rename columns
    df_result.rename(columns={col: f'{col}_list' for col in col_to_list}, inplace=True)
    # sort values
    df_result.sort_values(col_to_stay, inplace=True)
    # drop row with only nan value
    df_result.dropna(axis=0, how='all', inplace=True)
    # reset index
    df_result.reset_index(drop=True, inplace=True)
    return df_result

