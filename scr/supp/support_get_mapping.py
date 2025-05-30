
try:
    from supp.support_load import get_path
    from supp.support_load import read_csv
except:
    from support_load import get_path
    from support_load import read_csv


# map for sorted dynasty age
def get_dynasty_sorted():
    import numpy as np
    import pandas as pd
    file_name = 'map_dynasty_sorted.xlsx'
    path = get_path(file_name)
    return pd.read_excel(path)["dynasty_sorted"].to_numpy(dtype='str')


# dynasty age transformed into numbers
def get_dynasty_age():
    import pandas as pd
    file_name = 'map_dynasty_age.xlsx'
    path = get_path(file_name)
    return pd.read_excel(path, index_col="original_age", dtype='str')["aggregated_age"].to_dict()


# title category aggregation map
def get_category_aggregation():
    import pandas as pd
    file_name = 'map_category_aggregated.xlsx'
    path = get_path(file_name)
    return pd.read_excel(path, index_col="category", dtype='str')["aggregated_category"].to_dict()


# unpack list
def unpack_list(df, col_name):
    import ast
    # drop nan values
    df = df.dropna(subset=[col_name])
    # convert 'col_name' column from string to actual list
    df.loc[:, col_name] = df.loc[:, col_name].map(ast.literal_eval)
    # unpack lists into multiple rows, puts nan in list is empty
    df = df.explode(col_name, ignore_index=True)
    return df


# load corresponding map and apply it.
def apply_map(df, col_name, map_version_dict=dict()):
    # get map version if specified in 'map_version_dict'
    map_version = map_version_dict.get('col_name', '')
    # make 'file_name' of the .csv file
    file_name = rf'map_{col_name}{map_version}.csv'
    # load map into dictionary
    map_dict = read_csv(file_name, index_col="key", dtype='str')
    if map_dict is None:
        print(f'\tNo map found')
        print(f'\tsupport_get_mapping.py -> apply_map(df, col_name, map_version_dict=dict()')
        print(f'\tContinuing without map.')
        return df
    map_dict = map_dict['value'].to_dict()
    # get list of unmapped levels
    unmapped = list(set(df[col_name].unique()) - set(map_dict.keys()))
    if len(unmapped) > 0:
        print(f'\tUnmapped levels of column {col_name}:\t{unmapped}')
    # apply map, puts nan where the key has no match
    df[col_name] = df[col_name].map(map_dict)
    return df


# perform one got encoding
def one_hot_encoding(df, col_name):
    # assign(dummy=1) -> adds new columns of '1'
    # pivot_table -> new df with rows = 'person_ID', columns=levels of 'col_name'
    #   filling 1 if pair ('person_ID', levels of 'col_name') is in df, else 0
    df = df.assign(dummy=1).pivot_table(index='ID_person',
                                        columns=col_name,
                                        values='dummy',
                                        fill_value=0
                                        ).reset_index()
    return df

