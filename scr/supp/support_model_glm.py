import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd

try:
    from supp.support_load import read_csv, read_excel
except:
    from support_load import read_csv, read_excel


# Define map between GLM model name and this script file name
# model_name to file_name
mtof = {
    'y_step_glm': 'glm_vizier_vs_title_step',
    'y_lasso_glm': 'glm_vizier_vs_title_lasso',
    'y_ridge_glm': 'glm_vizier_vs_title_ridge'
}
ftom = {value: key for key, value in mtof.items()}


# Concat all titles and its coeff according to model
def get_model_coef_df(file_name=None, version=''):
    global mtof, ftom

    # Set file_name to excel file with all GLM summary
    if file_name is None:
        file_name = rf"summary_model_coefficients_v0{version}"

    # Load data with GLM summary
    dfs_coef = read_excel(file_name)

    key_mapping = {key: [value for value in mtof.values() if key.startswith(value)][0]
                   for key in dfs_coef.keys()
                   if key.startswith('glm_vizier_vs_title')}
    dfs_coef = {key_mapping.get(key, key): value for key, value in dfs_coef.items()}

    df_list = []
    for value in mtof.values():
        df = dfs_coef[value]
        df = df[['Variable', 'Estimate']].copy()
        # Rename column 'Estimate' to model_name (y_step_glm, ...)
        df.rename(columns={'Estimate': ftom[value]}, inplace=True)
        df_list.append(df)

    # Perform OUTER JOIN on title for all models
    df_model_coef = df_list[0]
    for df in df_list[1:]:
        df_model_coef = pd.merge(df_model_coef, df, on='Variable', how='outer')

    # Set title as index
    df_model_coef.set_index('Variable', inplace=True)

    # Count non-null values per row
    df_model_coef['non_nan_count'] = df_model_coef.notna().sum(axis=1)
    # Sort by non-NaN count (descending), then by 'A' (ascending)
    df_model_coef = df_model_coef.sort_values(by=['non_nan_count', *list(mtof.keys())], ascending=False)
    df_model_coef = df_model_coef.drop(columns=['non_nan_count'])

    # Transpose dataframe
    df_model_coef = df_model_coef.transpose()
    #df_model_coef.reset_index(drop=False, inplace=True, names=['model'])
    df_model_coef.index = df_model_coef.index.map(lambda x: x[2:])

    return df_model_coef


def get_model_coef_dict(model_name, df=None, file_name=None, version=''):
    if df is None:
        df = get_model_coef_df(file_name=file_name, version=version)

    d = {col: df.loc[model_name, col] for col in df.columns
         if not np.isnan(df.loc[model_name, col])}
    return d


# Define function that can be easy use for model prediction
def get_model_glm_forward(model_name, file_name=None, version='', df_coef=None):
    from scipy.special import expit  # Numerically stable sigmoid function

    # Get dictionary of betas (coefficients), the map is feature_mame: beta
    coef_dict = get_model_coef_dict(model_name,
                                    df=df_coef,
                                    file_name=file_name,
                                    version=version)

    # Extract features from GLM model
    coefficient_names = [key for key in coef_dict.keys()]
    feature_names = [col for col in coefficient_names if col != '(Intercept)']
    # Move intercept to first position
    coefficient_names.remove('(Intercept)')
    coefficient_names.insert(0, '(Intercept)')
    # Get coefficients
    coefficients = [coef_dict[col] for col in coefficient_names]

    # define forward function
    def model_forward(x):
        # if passed pandas dataframe, convert it to numpy
        if isinstance(x, pd.DataFrame):
            x = x.loc[:, feature_names]

        x = np.array(x)
        # Ensure X is 2D if a single row is passed
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Create a column of ones for the intercept
        intercept_column = np.ones((x.shape[0], 1))
        # Add intercept to input features
        x = np.hstack((intercept_column, x))

        # Compute linear combination
        logit = np.dot(x, coefficients)
        # Apply sigmoid function
        #probabilities = 1 / (1 + np.exp(-logit))
        probabilities = expit(logit)
        return probabilities

    return model_forward, feature_names
