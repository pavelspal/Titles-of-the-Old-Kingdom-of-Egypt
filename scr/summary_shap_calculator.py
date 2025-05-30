import os
import numpy as np
import pandas as pd
import shap

import supp.support_load as sl
from supp.support_load import load_pickle, get_name_dicts
from supp.support_save import save_df
from supp.support_model_glm import get_model_glm_forward
from supp.support_model_nn import get_model_nn_forward
from supp.support_constants import PATH_SHAP_VALUES

"""
 BIO: 
 This script computes SHAPs values for all models.
 - Model are pre-defined in 'df_model_dict'.
 - It loads models (both GLM and NN).
 - It computed SHAPs on train and test sets (for all persons).
 - For each model it saves result
   into file 'shap_values__{model_name}__{model_file_name}__{data_set}'.
 - If the file already exist before starting computation, the file will be skipped.
   To recalculated these files ones must first delete them.
"""


# Get file name for the shap values file
def get_shap_values_file_name(model_name, model_file_name, data_set):
    model_file_name = os.path.splitext(model_file_name)[0]
    f_name = f'shap_values__{model_name}__{model_file_name}__{data_set}'
    return f_name


# Get file name for the shap values file
def get_data_file_name(data_set=None, version=None):
    if data_set is None or version is None:
        raise ValueError("Params 'data_set' and 'version' can not be None.")
    f_name = f'df_vizier_{data_set}_{version}_2'
    return f_name


def load_data(data_set=None, version=None):
    if data_set is None or version is None:
        raise ValueError("Params 'data_set' and 'version' can not be None.")
    f_name = get_data_file_name(data_set, version)
    df = sl.read_csv(f_name).set_index('ID_person')
    return df


# Function for calculation SHAP values
def get_shap_values(model_name, params, method=None, use_train=False):
    if method is None:
        method = shap.Explainer

    # Get model forward function
    f = params['function']
    # Get model feature names
    feature_names = params['feature_names']
    # Define background data
    background_data = params['X_train']
    background_data = background_data[feature_names]
    # Extract the randomly selected rows
    indices = np.random.choice(background_data.shape[0], size=100, replace=False)
    background_data = background_data.iloc[indices, :]
    # Define data
    data = params['X_train'] if use_train else params['X_test']
    data = data[feature_names]

    # Explain the model using SHAP
    explainer = method(f, background_data)  # SHAP wrapper
    shap_values = explainer(data)

    # convert SHAP values to pandas dataframe
    if isinstance(shap_values, shap.Explanation):
        shap_values = shap_values.values
    shap_values = pd.DataFrame(shap_values, index=data.index, columns=feature_names)

    print(f'SHAP values for {model_name} calculated with {method}')
    return shap_values, explainer


# Must be SHAP values recalculated if file already exists?
force_recalculation = False
folder_shap = PATH_SHAP_VALUES

if __name__ == '__main__':
    # ------------------------------
    # LOAD SETS AND MODELS
    # ------------------------------
    df_model = sl.read_csv('df_model_dict')
    df_model = df_model.sort_values('order')
    # get list of all model names
    model_name_list = df_model['model_name'].to_list()
    file_name_dict = df_model.set_index('model_name')['file_name'].to_dict()

    # predefine dictionary for data and model function
    data_train_dict = {}
    data_val_dict = {}
    data_test_dict = {}
    model_forward_dict = {}
    feature_names_dict = {}

    # GLM models:
    df_glm = df_model.loc[df_model['model_type'] == 'glm', :]
    for index, row in df_glm.iterrows():
        model_name = row['model_name']
        print(f'\nLoading model {model_name}:')
        # load data
        version = row['feature_version']
        data_train_dict[model_name] = load_data(data_set='train', version=version)
        data_val_dict[model_name] = load_data(data_set='val', version=version)
        data_test_dict[model_name] = load_data(data_set='test', version=version)
        # load model forward
        file_name = row['file_name']
        function, feature_names = get_model_glm_forward(model_name[:-3], file_name=file_name)
        model_forward_dict[model_name] = function
        feature_names_dict[model_name] = feature_names

    # NN models
    df_nn = df_model.loc[df_model['model_type'] == 'nn', :]
    for index, row in df_nn.iterrows():
        model_name = row['model_name']
        print(f'\nLoading model {model_name}:')
        # load data
        version = row['feature_version']
        data_train_dict[model_name] = load_data(data_set='train', version=version)
        data_val_dict[model_name] = load_data(data_set='val', version=version)
        data_test_dict[model_name] = load_data(data_set='test', version=version)
        # load model forward
        folder = row['folder']
        file_name = row['file_name']
        path = os.path.join(folder, file_name)
        function, feature_names = get_model_nn_forward(path)
        model_forward_dict[model_name] = function
        feature_names_dict[model_name] = feature_names


    # ------------------------------
    # COMPUTE SHAPs VALUES
    # ------------------------------
    # define dictionary with parameters for each model
    model_params_dict = {model_name: {'function': model_forward_dict[model_name],
                                      'X_train': data_train_dict[model_name],
                                      'X_test': data_test_dict[model_name],
                                      'feature_names': feature_names_dict[model_name]
                                      }
                         for model_name in model_name_list}

    # COMPUTE SHAP VALUES FOR TEST SET
    print('\n\n------------------------------')
    print(f'COMPUTE SHAP VALUES FOR TEST SET')
    print('------------------------------')
    shaps_test = {}
    expleiners_test = {}
    for model_name in model_name_list:
        print(f'RUNNING {model_name}')
        # check whether file already exists
        file_name = get_shap_values_file_name(model_name, file_name_dict[model_name], 'test')
        print(f'\tfile_name: {file_name}')
        full_path = os.path.join(folder_shap, file_name + '.csv')
        if os.path.isfile(full_path) and not force_recalculation:
            print(f'\tFile already exists. Skipping this stage.')
            continue

        params = model_params_dict[model_name]
        shap_values, explainer = get_shap_values(model_name, params)
        shaps_test[model_name] = shap_values
        expleiners_test[model_name] = explainer
        # save SHAP values
        save_df(shap_values, file_name=file_name, folder=folder_shap, save_index=True)

    # COMPUTE SHAP VALUES FOR TRAIN SET
    print('\n\n------------------------------')
    print(f'COMPUTE SHAP VALUES FOR TRAIN SET')
    print('------------------------------')
    shaps_train = {}
    expleiners_train = {}
    for model_name in model_name_list:
        print(f'RUNNING {model_name}')
        # check whether file already exists
        file_name = get_shap_values_file_name(model_name, file_name_dict[model_name], 'train')
        print(f'\tfile_name: {file_name}')
        full_path = os.path.join(folder_shap, file_name + '.csv')
        if os.path.isfile(full_path) and not force_recalculation:
            print(f'\tFile already exists. Skipping this stage.')
            continue

        params = model_params_dict[model_name]
        shap_values, explainer = get_shap_values(model_name, params, use_train=True)
        shaps_train[model_name] = shap_values
        expleiners_train[model_name] = explainer
        # save SHAP values
        save_df(shap_values, file_name=file_name, folder=folder_shap, save_index=True)


    print('\n\nFINISHED')
