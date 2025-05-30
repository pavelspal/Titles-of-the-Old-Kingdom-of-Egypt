import pandas as pd

from supp.support_load import read_csv
from supp.support_save import save_df
from supp.support_constants import PATH_DATA, PATH_DESCRIPTION
from supp.support_get_mapping import unpack_list, apply_map, one_hot_encoding
from supp.support_analyzer import make_excel_analysis


"""
--------------------------------
    FILE BIO
--------------------------------
- file_name: unpack_df_all.py
- motivation:
    files such 'df_person_all' that for each person includes all records (mostly in list format),
    this script unpack such file into one hot encoding

"""

df_name = 'df_person_all'
col_name = None#'00_dynasty_list'
col_id = None #'08'

map_version_dict = {
    '08_sex': ''
}


# unpack column
def unpack_column(df_big, col_name):
    # select appropriate columns
    df = df_big[['ID_person', col_name]].copy()
    # if needed, unpack list
    if col_name[:2] != '08':
        df = unpack_list(df, col_name)
    # apply map
    df = apply_map(df, col_name, map_version_dict)
    # drop nan values
    df = df.dropna(subset=[col_name])
    # perform one hot encoding
    df = one_hot_encoding(df, col_name)
    # perform a left merge to keep all 'person_ID'
    df = pd.merge(df_big[['ID_person']], df, on='ID_person', how='left')
    # fill missing values in df columns with 0
    df.fillna(0, inplace=True)
    return df


df_original = read_csv(df_name)

column_list = None
if col_name is not None:
    column_list = [col_name]
elif col_id is not None:
    column_list = [col for col in df_original.columns if col[:2] == col_id]
else:
    column_list = [col for col in df_original.columns if col[:2].isdigit()]

for column in column_list:
    print('-----------------------------------')
    print(f'UNPACKING OF {column}')
    df_unpacked = unpack_column(df_original, column)
    save_df(df_unpacked, file_name=f'{column}', folder=f'{PATH_DATA}/unpacked/{df_name}')
    make_excel_analysis(df_unpacked, file_name=f'{column}', save_path=f'{PATH_DESCRIPTION}/unpacked/{df_name}')
    print(f'FINISHED UNPACKING OF {column}\n')


print('FINISHED')
