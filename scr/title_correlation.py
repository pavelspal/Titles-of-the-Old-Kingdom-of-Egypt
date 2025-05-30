import os
import numpy as np
import pandas as pd
import time

from supp.support_load import load_pickle, get_name_dicts
from supp.support_analyzer import make_excel
from supp.support_correlation import heat_map, split_titles, describe
from supp.support_get_mapping import get_category_aggregation


# path to separated sheets
save_path = r'plot_title_correlation/'
save_path_csv = r'data/'
# Check if the folder already exists
if not os.path.exists(save_path):
    # If not, create the folder
    os.makedirs(save_path[:-1])


# ------------------------------
# LOAD DATA
# ------------------------------
dfs, dfs_name, dfs_export_date = load_pickle()
# iton ... index to name dictionary
# ntoi ... name to index dictionary
iton, ntoi = get_name_dicts(dfs_name)


# ------------------------------
# PREPARE DATAFRAME
# ------------------------------
# select dataframe
df_general = dfs[ntoi['df_general']]
df_titles_general = dfs[ntoi['df_titles_general']]
df_title = dfs[ntoi['df_titles']]

# drop rows with any nan values
df_titles_general = df_titles_general[["ID_official", "ID_title"]]
print(f"df_titles_general, no. of nans = {df_titles_general.isnull().any(axis=1).sum()}")
df_titles_general.dropna(inplace=True)
# drop duplicated rows
print(f'df_titles_general, no. duplicates = {df_titles_general.duplicated().sum()}')
df_titles_general.drop_duplicates(inplace=True)
# due to nan values 'ID_title' was loaded as 'float':
try:
    df_titles_general['ID_title'] = df_titles_general['ID_title'].astype(np.int64)
except:
    print('ERROR. Can not convert given column into int.')
# print dtypes of each dataframe
#print(f'\tGENRERAL dtypes:\n{df_general.dtypes}')
#print(f'\tTITLE MAP dtypes:\n{df_title_map.dtypes}')
#print(f'\tTITLE dtypes:\n{df_title.dtypes}\n')


# ------------------------------
# MERGING DATAFRAMES
# ------------------------------
print('MERGE:')
# merge 'ID_official' with 'ID_person'
data = pd.merge(df_titles_general, df_general, on='ID_official', how='inner')
#data.to_csv("data/merge_df_titles_general_df_general_on_ID_official.csv")
data = data[['ID_person', 'ID_title']]
print(f'merge: df_title_map + df_general, no. duplicated = {data.duplicated().sum()}')
duplicated_1 = data[data.duplicated(keep=False)].sort_values(by=['ID_person', 'ID_title'])
data.drop_duplicates(inplace=True)
# merge 'ID_title' with 'type' of title
data = pd.merge(data, df_title, on='ID_title', how='inner')
data = data[['ID_person', 'type']]
print(f'merge: previous + df_title, no. duplicated = {data.duplicated().sum()}')
duplicated_2 = data[data.duplicated(keep=False)].sort_values(by=['ID_person', 'type'])
data.drop_duplicates(inplace=True)
# drop rows with any nan values
n_row_total = data.shape[0]
data.dropna(inplace=True)
n_row_non_nan = data.shape[0]
print(f'nan ratio of the final dataframe = {round(100-n_row_non_nan/n_row_total*100, 1)}')
titles_list = pd.DataFrame(data['type'].tolist())
titles_split = split_titles(data['type'].tolist())
# Reindex! for correct concat. Both dfs must have same index array!
data.reset_index(inplace=True, drop=True)
data = pd.concat([data, titles_split], axis=1)
# sort dataframe
data.sort_values(by=['type', 'category', 'sub_category'], inplace=True)
# reindex for convenience
data.reset_index(inplace=True, drop=True)
# make an Excel statistic about the final dataframe
# category to cluster
data = data[data['category'] != 'uncertain']
data['cluster'] = data['category'].map(get_category_aggregation())

make_excel(data, save_path + 'df_person_title_description', date=dfs_export_date)
# save data
data.to_csv('data/person_title.csv')


# ------------------------------
# ANALYZE DATAFRAME
# ------------------------------

# ENTIRE TITLES
print('\n\nENTIRE TITLES')
# make pivot table P of shape (n, m)
# n ... number of persons
# m ... number of titles
# (P)_ij = 1 ... if i-th person has j-th title
#        = 0 ... otherwise
pivot_table = data.pivot_table(index='ID_person', columns='type', aggfunc='size', fill_value=0)
# describe flattened pivot table -> is max value == 1? -> No duplicates
describe(pivot_table)

# make correlation matrix C
# C_ij = k ... where k is number of persons with both title i and j
#      = sum_{l=1 up to n} P_li * P_lj  ... person l must have both title i and j
#      = sum_{l=1 up to n} (P^T)_il * P_lj ... transposition of P_li
#      = (P^T @ P)_ij ... definition of matrix multiplication
correlation = pivot_table.T.dot(pivot_table)  # P^T @ P

# plot heat_map
heat_map(correlation, save_path=save_path, name='all titles')


# TITLE CATEGORY
print('\n\nTITLE CATEGORY')
encoding_to = 'cluster'
data_job = data[['ID_person', encoding_to]]
pivot_table_counts = data_job.pivot_table(index='ID_person', columns=encoding_to, aggfunc='size', fill_value=0)
pivot_table_counts.to_csv(f'data/person_title_{encoding_to}_pivot.csv')
data_job.drop_duplicates(inplace=True)
pivot_table = data_job.pivot_table(index='ID_person', columns=encoding_to, aggfunc='size', fill_value=0)
pivot_table.to_csv(f'data/person_title_{encoding_to}_ohe.csv')
describe(pivot_table)
# make correlation matrix C
correlation = pivot_table.T.dot(pivot_table)  # P^T @ P
# plot heat_map
heat_map(correlation, save_path=save_path, name=f'{encoding_to} titles')


# # TITLE SUBCATEGORIES
# print('\n\nTITLES SUBCATEGORIES')
# unique_categories = data['category'].unique()
# for category in unique_categories:
#     print(f'\n\t{category}')
#     data_subcat = data[data['category'] == category]
#     data_subcat = data_subcat[['ID_person', 'sub_category']].drop_duplicates()
#     pivot_table = data_subcat.pivot_table(index='ID_person', columns='sub_category', aggfunc='size', fill_value=0)
#     #describe(pivot_table)
#     # make correlation matrix C
#     correlation = pivot_table.T.dot(pivot_table)  # P^T @ P
#     # plot heat_map
#     heat_map(correlation, save_path=save_path, name=f'{category} titles')
#     # Pause execution for 3 seconds
#     time.sleep(3)


print('\n\nFINISHED')
