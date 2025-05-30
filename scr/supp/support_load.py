
try:
    from supp.support_constants import *
    from supp.support_mbukacek_fce import merge_on_jones
    from supp.support_load_preprocessing import remove_jones_duplicates
except:
    from support_constants import *
    from support_mbukacek_fce import merge_on_jones
    from support_load_preprocessing import remove_jones_duplicates


# project_root/
# ├── data/
# │   └── data.csv
# └── supp/
# │   └── supp_script.py
# └── script_1.py
# └── script_2.py
# transform path to this_script.py into path to data.csv
# -> go one folder back and move into folder 'data'
def get_path(file_name, folder=PATH_DATA):
    import os
    concurrent_dir = os.path.dirname(__file__)
    path = os.path.join(concurrent_dir, '..', folder, file_name)
    path = os.path.normpath(path)
    return path

def read_excel(file_name, folder=PATH_DATA, **kwargs):
    """
    Reads a Excel file into a list of pandas DataFrames.

    Parameters:
        file_name (str): The name of the CSV file.
        **kwargs: Optional keyword arguments to pass to pandas.read_excel().

    Returns:
        pd.DataFrame: The list of DataFrames.
    """
    import pandas as pd
    import os

    # add '.csv' ending if needed
    if not file_name.endswith(".xlsx"):
        file_name += '.xlsx'
    # find path
    path = get_path(file_name, folder=folder)
    if not os.path.isfile(path):
        path = get_path(file_name, folder=PATH_DATA)
    if not os.path.isfile(path):
        path = get_path(file_name, folder=PATH_DATA_MAP)
    if not os.path.isfile(path):
        path = get_path(file_name, folder=PATH_DATA_UNPACT)
        path = get_path(file_name, folder=PATH_SHAP_VALUES)
    if not os.path.isfile(path):
        path = get_path(file_name, folder=PATH_SHAP_ABS_MEAN)
    if not os.path.isfile(path):
        print('File not found.')
        print('\n'.join([folder, PATH_DATA, PATH_DATA_MAP, PATH_DATA_UNPACT]))
        return None

    try:
        dfs = pd.read_excel(path, sheet_name=None, **kwargs)  # None loads all sheets
        print('Excel file loaded.')
        print(f"{path}")
        return dfs
    except Exception as e:
        print(f"Error reading the Excel file: {e}\n{path}")
        return None

def read_csv(file_name, folder=PATH_DATA, **kwargs):
    """
    Reads a CSV file into a pandas DataFrame.

    Parameters:
        file_name (str): The name of the CSV file.
        **kwargs: Optional keyword arguments to pass to pandas.read_csv().

    Returns:
        pd.DataFrame: The DataFrame containing the CSV data.
    """
    import pandas as pd
    import os

    # add '.csv' ending if needed
    if not file_name.endswith(".csv"):
        file_name += '.csv'
    # find path
    path = get_path(file_name, folder=folder)
    if not os.path.isfile(path):
        path = get_path(file_name, folder=PATH_DATA)
    if not os.path.isfile(path):
        path = get_path(file_name, folder=PATH_DATA_MAP)
    if not os.path.isfile(path):
        path = get_path(file_name, folder=PATH_DATA_UNPACT)
    if not os.path.isfile(path):
        path = get_path(file_name, folder=PATH_DATA_MERGED)
    if not os.path.isfile(path):
        path = get_path(file_name, folder=PATH_SHAP_VALUES)
    if not os.path.isfile(path):
        path = get_path(file_name, folder=PATH_SHAP_ABS_MEAN)
    if not os.path.isfile(path):
        print('File not found.')
        print('\n'.join([folder, PATH_DATA, PATH_DATA_MAP, PATH_DATA_UNPACT]))
        return None

    try:
        df = pd.read_csv(path, **kwargs)
        print('CSV file loaded.')
        print(f"{path}")
        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}\n{path}")
        return None


# open pickle file
def load_pickle():
    import pickle

    # path to pickle file
    folder = PATH_DATA
    file_name = 'dfs_complete.pickle'
    path = get_path(file_name, folder=folder)
    try:
        with open(path, 'rb') as f:
            data_loaded = pickle.load(f)
            # separating into three parts
            dfs = data_loaded[0]  # dataframes
            dfs_name = data_loaded[1]  # their names
            dfs_export_date = data_loaded[2]  # date (as a string) when the maatbase export was created by Veronika
            print('Pickle database loaded.')
            print(f"{path}")

            # apply preprocessing
            dfs, dfs_name, dfs_export_date = pickle_preprocessing(dfs, dfs_name, dfs_export_date)

            return [dfs, dfs_name, dfs_export_date]
    except:
        print(f'ERROR. Pickle database is not in the path {path}.')
    return None


def get_name_dicts(dfs_name):
    # dictionary: index of a table -> the name of the table
    iton = {dfs_name.index(name): name for name in dfs_name}
    # dictionary: name of a table -> index of the table
    ntoi = {name: index for index, name in iton.items()}
    return [iton, ntoi]


# apply preprocessing to data
def pickle_preprocessing(dfs, dfs_name, dfs_export_date):
    # get dictionary
    iton, ntoi = get_name_dicts(dfs_name)
    # get df_titles and df_person_title
    df_titles = dfs[ntoi['df_titles']]
    df_person_title = dfs[ntoi['df_person_title']]

    # apply merge_on_jones from M. Bukacek
    df_titles_new, df_person_title_new = merge_on_jones(df_titles, df_person_title)
    # rewrite old df_titles and df_person_title
    dfs[ntoi['df_titles']] = df_titles_new
    dfs[ntoi['df_person_title']] = df_person_title_new
    # print information about preprocessing
    print(f'Applied preprocessing: merge_on_jones')

    # remove duplicated Jones id
    df_titles_new, df_person_title_new = remove_jones_duplicates(df_titles_new, df_person_title_new)
    # rewrite old df_titles and df_person_title
    dfs[ntoi['df_titles']] = df_titles_new
    dfs[ntoi['df_person_title']] = df_person_title_new
    # print information about preprocessing
    print(f'Applied preprocessing: remove_jones_duplicates')

    return [dfs, dfs_name, dfs_export_date]


