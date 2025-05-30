import numpy as np
import pandas as pd
import os

import supp.support_load as sl
from supp.support_load import load_pickle, get_name_dicts
from supp.support_save import save_df
from supp.support_model_glm import get_model_glm_forward
from supp.support_model_nn import get_model_nn_forward

"""
 BIO: 
 This script summarizes persons probabilities from all models.
 - Model are pre-defined in 'df_model_dict'.
 - It loads models (both GLM and NN).
 - It computed predictions on train, validation and test sets (for all persons).
 - It calculates statistics for each person and its performance in different models.
 - It load person name from database (if name available).
 - It save results into csv file 'summary_person'.
"""


if __name__ == '__main__':
    # ------------------------------
    # LOAD DATA
    # ------------------------------
    dfs, dfs_name, dfs_export_date = load_pickle()
    # iton ... index to name dictionary
    # ntoi ... name to index dictionary
    iton, ntoi = get_name_dicts(dfs_name)


    # ------------------------------
    # LOAD SETS AND MODELS
    # ------------------------------
    df_model = sl.read_csv('df_model_dict')
    df_model = df_model.sort_values('order')
    # get list of all model names
    model_name_list = df_model['model_name'].to_list()

    # predefine dictionary for data and model function
    data_dict = {}
    model_forward_dict = {}

    # GLM models:
    df_glm = df_model.loc[df_model['model_type'] == 'glm', :]
    for index, row in df_glm.iterrows():
        model_name = row['model_name']
        print(f'\nLoading model {model_name}:')
        # load data
        version = row['feature_version']
        df_train = sl.read_csv(f'df_vizier_train_{version}_2')
        df_val = sl.read_csv(f'df_vizier_val_{version}_2')
        df_test = sl.read_csv(f'df_vizier_test_{version}_2')
        df_data = pd.concat([df_train, df_val, df_test])
        data_dict[model_name] = df_data
        # load model forward
        file_name = row['file_name']
        function, _ = get_model_glm_forward(model_name[:-3], file_name=file_name)
        model_forward_dict[model_name] = function

    # NN models
    df_nn = df_model.loc[df_model['model_type'] == 'nn', :]
    for index, row in df_nn.iterrows():
        model_name = row['model_name']
        print(f'\nLoading model {model_name}:')
        # load data
        version = row['feature_version']
        df_train = sl.read_csv(f'df_vizier_train_{version}_2')
        df_val = sl.read_csv(f'df_vizier_val_{version}_2')
        df_test = sl.read_csv(f'df_vizier_test_{version}_2')
        df_data = pd.concat([df_train, df_val, df_test])
        data_dict[model_name] = df_data
        # load model forward
        folder = row['folder']
        file_name = row['file_name']
        path = os.path.join(folder, file_name)
        function, _ = get_model_nn_forward(path)
        model_forward_dict[model_name] = function


    # ------------------------------
    # MAKE PREDICTIONS
    # ------------------------------
    prediction_list = []
    for model_name in model_name_list:
        df_test = data_dict[model_name]
        function = model_forward_dict[model_name]
        # make predictions
        pred = function(df_test).flatten()
        d = {'ID_person': df_test['ID_person'].to_numpy(), model_name: pred}
        df_pred = pd.DataFrame(d)
        prediction_list.append(df_pred)

    df_predictions = prediction_list[0]
    for df_pred in prediction_list[1:]:
        df_predictions = pd.merge(df_predictions, df_pred, on='ID_person', how='inner')


    # ------------------------------
    # CALC STATISTICS ABOUT PREDICTIONS
    # ------------------------------
    prediction_columns = [col for col in df_predictions.columns if col != 'ID_person']
    df_predictions.set_index('ID_person', inplace=True)
    df_predictions['mean'] = df_predictions.mean(axis=1)
    df_predictions['std'] = df_predictions.std(axis=1)
    df_predictions['max'] = df_predictions.max(axis=1)
    df_predictions['min'] = df_predictions.min(axis=1)
    df_predictions['max-min'] = df_predictions['max'] - df_predictions['min']
    df_predictions.reset_index('ID_person', inplace=True)
    # rearrange column order
    stat_cols = ['mean', 'std', 'max', 'min', 'max-min']
    other_cols = [col for col in df_predictions.columns if col not in stat_cols]
    df_predictions = df_predictions[stat_cols + other_cols]


    # ------------------------------
    # GET INFO ABOUT PERSONS
    # ------------------------------
    id_person_list = df_predictions['ID_person']
    df_general = dfs[ntoi['df_general']]
    df_name = dfs[ntoi['df_name']]
    # get persons names
    df_person_info = pd.merge(df_general, df_name,
                              left_on='ID_official', right_on='ID_official_source', how='inner')
    df_person_info = df_person_info[['ID_person', 'name']].sort_values(['ID_person', 'name'])
    df_person_info = df_person_info.drop_duplicates(subset=['ID_person'])
    df_person_info = df_person_info[df_person_info['ID_person'].isin(id_person_list)]
    # set vizier
    df_vizier = data_dict[list(data_dict.keys())[0]][['ID_person', 'vizier']]
    df_person_info = pd.merge(df_person_info, df_vizier, on='ID_person', how='right')
    # set persons set
    id_train = sl.read_csv(f'df_vizier_train_v1_2')['ID_person']
    id_val = sl.read_csv(f'df_vizier_val_v1_2')['ID_person']
    id_test = sl.read_csv(f'df_vizier_test_v1_2')['ID_person']
    df_person_info['set'] = 'Unknown'
    df_person_info.loc[df_person_info['ID_person'].isin(id_train), 'set'] = 'Train'
    df_person_info.loc[df_person_info['ID_person'].isin(id_val), 'set'] = 'Validation'
    df_person_info.loc[df_person_info['ID_person'].isin(id_test), 'set'] = 'Test'


    # ------------------------------
    # SAVE SUMMARY
    # ------------------------------
    df_summary_person = pd.merge(df_person_info, df_predictions, on='ID_person', how='right')
    save_df(df_summary_person, file_name='summary_person')


    print('\n\nFINISHED')
