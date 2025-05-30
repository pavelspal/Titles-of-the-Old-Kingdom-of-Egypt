import os
import numpy as np
import pandas as pd

import supp.support_load as sl
from supp.support_plots import *
from supp.support_constants import CONTINUOUS_FEATURES
from summary_shap_calculator import load_data



omit_features = CONTINUOUS_FEATURES

if __name__ == '__main__':
    # ------------------------------
    # LOAD SETS AND MODELS
    # ------------------------------
    df_model = sl.read_csv('df_model_dict')
    df_model = df_model.sort_values('order')
    # get list of all model names
    model_name_list = df_model['model_name'].to_list()


    # ------------------------------
    # LOAD SHAP MEAN VALUES
    # ------------------------------
    df_shap_train = sl.read_csv('df_shap_train').set_index('title')
    df_shap_test = sl.read_csv('df_shap_test').set_index('title')
    # Omit features
    keep_features = [ind for ind in df_shap_train.index if ind not in omit_features]
    df_shap_train = df_shap_train.loc[keep_features, :]
    df_shap_test = df_shap_test.loc[keep_features, :]

    # ------------------------------
    # LOAD PLOT SHAP MEAN
    # ------------------------------
    for model_name in model_name_list:
        # SHAP Absolute Mean Plots
        # Separately train and test set
        s_shap = df_shap_test[model_name]
        plot_shap_mean(s_shap, model_name, data_set='Test')


    # ------------------------------
    # LOAD PLOT SHAP MEAN CONCAT
    # ------------------------------
    for model_name in model_name_list:
        # SHAP Absolute Mean Plots
        # Separately train and test set
        s_shap_train = df_shap_train[model_name]
        s_shap_test = df_shap_test[model_name]
        shap_dict = {
            'train': s_shap_train,
            'test': s_shap_test
        }
        plot_shap_mean_concat(shap_dict, model_name)

    print('\n\nFINISHED')
