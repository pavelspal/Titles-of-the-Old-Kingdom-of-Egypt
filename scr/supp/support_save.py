try:
    from supp.support_constants import *
    from supp.support_load import get_path
except:
    from support_constants import *
    from support_load import get_path




# this files save data into folder 'data' with most of dataframes
def save_df(df, file_name, folder=PATH_DATA, save_index=False):
    import os
    import pandas as pd
    try:
        # get path where to save the df
        path = get_path(file_name, folder=folder)
        # ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # hardcode format to CSV
        path = os.path.splitext(path)[0] + '.csv'
        # save the df
        df.to_csv(path, index=save_index)
        print(f"Dataframe saved into {path}")
    except Exception as e:
        print(f"Error saving file: {e}")
    return None
    
