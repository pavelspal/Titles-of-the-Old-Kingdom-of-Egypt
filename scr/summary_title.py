import os
import numpy as np
import pandas as pd
import shap

import supp.support_load as sl
import supp.support_title_stats as sts
from supp.support_save import save_df
from supp.support_constants import PATH_SHAP_VALUES, PATH_SHAP_ABS_MEAN, CONTINUOUS_FEATURES
from summary_shap_calculator import get_shap_values_file_name, load_data

"""
 BIO: 
 This script summarizes titles from all models.
 - Model are pre-defined in 'df_model_dict'.
 - It loads pre-computed SHAPs values.
 - It computes SHAP Absolute Mean value for each model feature.
 - It calculates statistics for each title.
 - It load title translation and other info from database.
 - It save results into csv file 'summary_title'.
"""


# Define path to shap values dataframes
folder_shap = PATH_SHAP_VALUES
folder_save = PATH_SHAP_ABS_MEAN
data_set_list = ['train', 'test']
# Define continuous features
continuous_features = CONTINUOUS_FEATURES


# Get file name for the shap abs mean file
def get_shap_abs_mean_file_name(model_name, model_file_name, data_set):
    model_file_name = os.path.splitext(model_file_name)[0]
    f_name = f'shap_abs_mean__{model_name}__{model_file_name}__{data_set}'
    return f_name


def get_shap_mean(shap_values, data, category=1, use_abs=False):
    """
    Custom SHAP summary plot for binary features.
    Plots only SHAP values where corresponding feature values match `category`.

    :param shap_values: DataFrame SHAP values (same shape as `data`)
    :param data: DataFrame containing features
    :param category: Value (0 or 1) to filter features for plotting (default=1)
    """

    data = data.loc[:, shap_values.columns]
    # Check consistency of shap_values and data
    if shap_values.shape != data.shape:
        raise ValueError("Dataframes 'shap_values' and 'data' must have same shape.")
    if not shap_values.columns.equals(data.columns):
        raise ValueError("Dataframes 'shap_values' and 'data' must have same columns.")
    if not shap_values.index.equals(data.index):
        raise ValueError("Dataframes 'shap_values' and 'data' must have same index.")
    # Validate category input
    if category not in [0, 1]:
        raise ValueError("Category must be 0 or 1.")

    # Mask features where values match the category
    mask = (data == category)

    # Filter SHAP values
    filtered_shap = shap_values * mask

    # Compute mean absolute SHAP values per feature
    if use_abs:
        shap_abs_mean = filtered_shap.abs().sum(axis=0) / mask.sum(axis=0)
    else:
        shap_abs_mean = filtered_shap.sum(axis=0) / mask.sum(axis=0)

    # Compute SHAP abs mean values for continuous feature in standard way
    continuous_columns = [col for col in shap_abs_mean.index if col in continuous_features]
    for col in continuous_columns:
        if use_abs:
            shap_abs_mean[col] = shap_values[col].abs().mean()
        else:
            shap_abs_mean[col] = shap_values[col].mean()

    # Set index and column name
    shap_abs_mean.index.name = 'feature_name'
    shap_abs_mean.name = 'shap_abs_mean'
    # convert pandas series to pandas dataframe
    shap_abs_mean = pd.DataFrame(shap_abs_mean)

    return pd.DataFrame(shap_abs_mean)


# Get info about titles
def get_title_info(col_titles_info=None):
    dfs, dfs_name, dfs_export_date = sl.load_pickle()
    iton, ntoi = sl.get_name_dicts(dfs_name)
    df_titles = dfs[ntoi['df_titles']]

    if col_titles_info is None:
        col_titles_info = ['ID_title',
                           'title',
                           'translation_of_title',
                           'type',
                           'general_work_classification',
                           'recipient',
                           'specific_profession']

    return df_titles.loc[:, col_titles_info]


def get_title_stat(df_model=None):
    # Define sets and versions
    set_list = ['train', 'val', 'test']
    version_list = ['v1', 'v2']
    if df_model is not None:
        version_list = df_model['feature_version'].unique().tolist()

    # Load data
    df_list = []
    for version in version_list:
        dfs = [load_data(data_set=data_set, version=version) for data_set in set_list]
        df = pd.concat(dfs, axis=0)
        df_list.append(df)

    # Get all possible features
    df_all = df_list[0].copy()
    for df in df_list[1:]:
        missing_features = [col for col in df.columns if col not in df_all.columns]
        df_all = pd.merge(df_all, df[missing_features], left_index=True, right_index=True, how='inner')
    # Check final shape
    if df_all.shape[0] != df_list[0].shape[0]:
        raise ValueError("ERROR. Method: 'get_title_stat', file: 'summary_shap_abs'. Different shapes after merge.")

    # Compute stats for title
    columns_to_ommit = continuous_features
    title_list = [col for col in df_all.columns if col not in columns_to_ommit]
    stat_list = []
    for title in title_list:
        d = {
             'title': title,
             'count': sts.get_count(df_all, title),
             'P(vizier|title)': sts.conditional_probability(df_all, 'vizier', title),
             'P(title|vizier)': sts.conditional_probability(df_all, title, 'vizier'),
        }
        stat_list.append(d)
    df_title_stat = pd.DataFrame(stat_list)

    return df_title_stat


# Concat SHAPs dfs
def get_df_shaps_concat(shap_dict):
    # Rename columns and collect
    renamed_dfs = [df.rename(columns={df.columns[0]: key}) for key, df in shap_dict.items()]
    # Outer join all on index
    result = pd.concat(renamed_dfs, axis=1, join='outer')
    return result


if __name__ == '__main__':
    # ------------------------------
    # LOAD SETS AND MODELS
    # ------------------------------
    df_model = sl.read_csv('df_model_dict')
    df_model = df_model.sort_values('order')
    # get list of all model names
    model_name_list = df_model['model_name'].to_list()
    file_name_dict = df_model.set_index('model_name')['file_name'].to_dict()


    # ------------------------------
    # COMPUTE SHAP ABS MEAN FOR EACH MODEL AND SET
    # ------------------------------
    shap_abs_mean_dict = {}
    for index, row in df_model.iterrows():
        model_name = row['model_name']

        d_shaps = {}
        for data_set in data_set_list:
            # Load SHAP values
            shap_file_name = get_shap_values_file_name(model_name, file_name_dict[model_name], data_set)
            df_shap = sl.read_csv(shap_file_name).set_index('ID_person')
            # Load train SHAP values
            version = row['feature_version']
            data = load_data(data_set=data_set, version=version)
            # Compute SHAP abs mean
            df_shap_abs_mean = get_shap_mean(df_shap, data)
            # Save SHAP abs mean
            file_name = get_shap_abs_mean_file_name(model_name, file_name_dict[model_name], data_set)
            save_df(df_shap_abs_mean, file_name=file_name, folder=folder_save, save_index=True)
            # save SHAPS abs mean and data into dict
            d_shaps[data_set] = df_shap_abs_mean
        shap_abs_mean_dict[model_name] = d_shaps


    # ------------------------------
    # CONCAT ALL SHAP ABS MEAN
    # ------------------------------
    shap_dict_train = {key: df_dict['train'] for key, df_dict in shap_abs_mean_dict.items()}
    shap_dict_test = {key: df_dict['test'] for key, df_dict in shap_abs_mean_dict.items()}

    df_shap_train = get_df_shaps_concat(shap_dict_train).reset_index()
    df_shap_train = df_shap_train.rename(columns={"feature_name": "title"})
    df_shap_test = get_df_shaps_concat(shap_dict_test).reset_index()
    df_shap_test = df_shap_test.rename(columns={"feature_name": "title"})
    df_title_info = get_title_info()
    df_title_stat = get_title_stat()

    # merge all dataframes
    df_summary_title = df_shap_test.sort_values(by=df_shap_test.columns.to_list()[1:], ascending=True)
    df_summary_title = pd.merge(df_summary_title, df_title_info, on='title', how='left')
    df_summary_title = pd.merge(df_title_stat, df_summary_title, on='title', how='right')
    # Move colum ID_title at the beginning
    df_summary_title = df_summary_title.set_index('ID_title').reset_index()
    df_summary_title["ID_title"] = df_summary_title["ID_title"].astype('Int64')

    # Save summary file
    save_df(df_shap_train, file_name='df_shap_train')
    save_df(df_shap_test, file_name='df_shap_test')
    save_df(df_summary_title, file_name='summary_title')


    print('\n\nFINISHED')
