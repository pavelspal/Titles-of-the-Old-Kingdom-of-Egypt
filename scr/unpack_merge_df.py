import pandas as pd
import os
import re

from supp.support_load import read_csv, get_path
from supp.support_save import save_df
from supp.support_constants import PATH_DATA, PATH_DESCRIPTION, VIZIER_IDS
from supp.support_get_mapping import unpack_list, apply_map, one_hot_encoding
from supp.support_analyzer import make_excel_analysis


"""
--------------------------------
    FILE BIO
--------------------------------
- file_name: unpack_merge_df.py
- motivation:
    folder r'data/unpack/{df_name}_all' contains dfs with all one hot encoded variables,
    this script merge these dfs with (explanatory variable) with response variable
"""


df_name = 'df_person_all'
response_variable_f_name = 'vizier'
folder = rf'{PATH_DATA}/unpacked/{df_name}'

df_response = read_csv(response_variable_f_name, folder=folder)

# find all csv files that start by 'XY', where 'X', 'Y' are digits
path = get_path(file_name='', folder=folder)
pattern = re.compile(r'^\d{2}.*\.csv$')
explanatory_file_list = [file for file in os.listdir(path) if pattern.match(file)]


# load dfs
for file in explanatory_file_list:
    print(f'--------------------------------------')
    print(f'merging file: {file}')
    df = read_csv(file, folder=folder)
    df_merged = df_response.merge(df, on='ID_person', how='inner')

    if df_response.shape[0] != df_merged.shape[0]:
        print(f'\tERROR shape does not match after merge.\n\tbefore: {df_response.shape[0]} \tafter: {df_merged.shape[0]}')

    # same and make excel description
    file_name = f'{response_variable_f_name}__{os.path.basename(file)}'
    save_df(df_merged, file_name=file_name, folder=f'{PATH_DATA}/unpacked/{df_name}/merged')
    make_excel_analysis(df_merged, file_name=file_name, save_path=f'{PATH_DESCRIPTION}/unpacked/{df_name}/merged')


print('FINISHED')
