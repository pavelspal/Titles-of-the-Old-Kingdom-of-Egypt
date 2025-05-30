import pandas as pd

from supp.support_load import read_csv
from supp.support_save import save_df
from supp.support_constants import PATH_DATA, PATH_DESCRIPTION, VIZIER_IDS
from supp.support_get_mapping import unpack_list, apply_map, one_hot_encoding
from supp.support_analyzer import make_excel_analysis


"""
--------------------------------
    FILE BIO
--------------------------------
- file_name: unpack_vizier.py
- motivation:
    file 'df_person_all' includes for each person list of all title_id 
    this script unpack this file and and map viziers via one hot encoding
"""

df_name = 'df_person_all'
col_name = 'ID_title_list'
vizier_title_id_list = VIZIER_IDS


print('-----------------------------------')
print(f'UNPACKING OF VIZIER')

# read file with all data corresponding to person
df_original = read_csv(df_name)

# select appropriate columns
df = df_original[['ID_person', col_name]].copy()
# unpack list
df = unpack_list(df, col_name)
# drop nan values
df = df.dropna(subset=[col_name])
# convert float to int
df[col_name] = df[col_name].astype(int)
# perform one hot encoding
df = one_hot_encoding(df, col_name)
# perform a left merge to keep all 'person_ID'
df = pd.merge(df_original[['ID_person']], df, on='ID_person', how='left')
# fill missing values in df columns with 0
df.fillna(0, inplace=True)
# convert float to int
df = df.astype(int)
# map if the person have any title_id corresponding to vizier
df['vizier'] = (df.loc[:, df.columns.isin(vizier_title_id_list)].sum(axis=1) > 0).astype(int)
# select only appropriate columns
df = df[['ID_person', 'vizier']]

# same and make excel description
save_df(df, file_name=f'vizier', folder=f'{PATH_DATA}/unpacked/{df_name}')
make_excel_analysis(df, file_name=f'vizier', save_path=f'{PATH_DESCRIPTION}/unpacked/{df_name}')


print(f'FINISHED UNPACKING OF VIZIER\n')
